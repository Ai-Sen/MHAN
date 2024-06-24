import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import GATConv, edge_softmax
from model.modelSelfAttention import SelfAttention

class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']

class HGTLayer(nn.Module):
    def __init__(self, in_dim, out_dim, node_dict, edge_dict, n_heads, dropout=0.2, use_norm=False):
        super(HGTLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.num_types = len(node_dict)
        self.num_relations = len(edge_dict)
        self.total_rel = self.num_types * self.num_relations * self.num_types
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)

        self.k_linears = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(self.num_types)])
        self.q_linears = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(self.num_types)])
        self.v_linears = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(self.num_types)])
        self.a_linears = nn.ModuleList([nn.Linear(out_dim, out_dim) for _ in range(self.num_types)])
        self.norms = nn.ModuleList([nn.LayerNorm(out_dim) for _ in range(self.num_types)]) if use_norm else None

        self.relation_pri = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        self.relation_att = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.skip = nn.Parameter(torch.ones(self.num_types))
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h):
        with G.local_scope():
            for srctype, etype, dsttype in G.canonical_etypes:
                sub_graph = G[srctype, etype, dsttype]
                k_linear = self.k_linears[self.node_dict[srctype]]
                v_linear = self.v_linears[self.node_dict[srctype]]
                q_linear = self.q_linears[self.node_dict[dsttype]]

                k = k_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                v = v_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                q = q_linear(h[dsttype]).view(-1, self.n_heads, self.d_k)

                e_id = self.edge_dict[etype]
                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]

                k = torch.einsum("bij,ijk->bik", k, relation_att)
                v = torch.einsum("bij,ijk->bik", v, relation_msg)

                sub_graph.srcdata["k"] = k
                sub_graph.dstdata["q"] = q
                sub_graph.srcdata["v_%d" % e_id] = v

                sub_graph.apply_edges(fn.v_dot_u("q", "k", "t"))
                attn_score = (sub_graph.edata.pop("t").sum(-1) * relation_pri / self.sqrt_dk)
                attn_score = edge_softmax(sub_graph, attn_score, norm_by="dst")
                sub_graph.edata["t"] = attn_score.unsqueeze(-1)

            G.multi_update_all(
                {etype: (fn.u_mul_e("v_%d" % e_id, "t", "m"), fn.sum("m", "t")) for etype, e_id in self.edge_dict.items()},
                cross_reducer="mean",
            )

            new_h = {}
            for ntype in G.ntypes:
                n_id = self.node_dict[ntype]
                alpha = torch.sigmoid(self.skip[n_id])
                t = G.nodes[ntype].data["t"].view(-1, self.out_dim)
                trans_out = self.drop(self.a_linears[n_id](t))
                trans_out = trans_out * alpha + h[ntype] * (1 - alpha)
                new_h[ntype] = self.norms[n_id](trans_out) if self.norms else trans_out
            return new_h

class HGT(nn.Module):
    def __init__(self, G, node_dict, edge_dict, n_inp, n_hid, n_out, n_layers, n_heads, use_norm=True):
        super(HGT, self).__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.gcs = nn.ModuleList()
        self.adapt_ws = nn.ModuleList([nn.Linear(n_inp, n_hid) for _ in range(len(node_dict))])
        for _ in range(n_layers):
            self.gcs.append(HGTLayer(n_hid, n_hid, node_dict, edge_dict, n_heads, use_norm=use_norm))
        self.linear = nn.Linear(n_hid, n_out)

    def forward(self, G):
        h = {ntype: F.gelu(self.adapt_ws[self.node_dict[ntype]](G.nodes[ntype].data["feature"])) for ntype in G.ntypes}
        for i in range(self.n_layers):
            h = self.gcs[i](G, h)
        return {ntype: self.linear(h[ntype]) for ntype in G.ntypes}

class MHAN(nn.Module):
    def __init__(self, G, in_size, hidden_size, out_size, n_layers, n_heads, n_fuse_heads, use_norm=True):
        super().__init__()
        node_dict = {ntype: i for i, ntype in enumerate(G.ntypes)}
        edge_dict = {etype: i for i, etype in enumerate(G.etypes)}
        for etype in G.etypes:
            G.edges[etype].data["id"] = torch.ones(G.number_of_edges(etype), dtype=torch.long).to(G.device) * edge_dict[etype]

        self.sage = HGT(G, node_dict, edge_dict, in_size, hidden_size, out_size, n_layers, n_heads, use_norm)
        self.pred = HeteroDotProductPredictor()
        self.gat = GATConv(in_size, out_size, num_heads=n_heads)
        self.selfAttention = SelfAttention(n_fuse_heads, out_size, out_size, 0.5)

    def forward(self, g, g2, neg_g, etype):
        h = self.sage(g)
        h2 = self.gat(g2, g2.ndata['feature'])
        h2 = torch.mean(h2, dim=1)
        z = torch.stack([h['NCT'], h2], dim=1)
        h['NCT'] = self.selfAttention(z)[:, 0, :]

        return self.pred(g, h, etype), self.pred(neg_g, h, etype)
