import argparse
import csv
import json
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import dgl
from sklearn.metrics import roc_curve, auc
from util.hitsScore import hits_score
from model.MHAN import MHAN
from data.dataloader import DataLoader
from util.ndcgScore import ndcg_score

timestamp = time.strftime("%Y%m%d-%H%M%S")
folder_name = f"./train_log/{timestamp}"

# 创建文件夹
os.makedirs(folder_name, exist_ok=True)

# 设置日志文件名
filename = f"{folder_name}/log.txt"

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def constructDGLGraph(dl, similarity, isRandomFeature):
    rawGraph = dl.generateRawTrainGraph()
    trailGraph = dl.generateTrailGraph(similarity)
    hetero_graph = dgl.heterograph({
        ('NCT', 'hasProblem', 'Problem'): (rawGraph['NCTP'], rawGraph['Problem']), 
        ('NCT', 'hasIntervention', 'Intervention'): (rawGraph['NCTI'], rawGraph['Intervention']),
        ('NCT', 'hasOutcome', 'Outcome'): (rawGraph['NCTO'], rawGraph['Outcome']),
        ('Problem', 'hasNCT1', 'NCT'): (rawGraph['Problem'], rawGraph['NCTP']),  
        ('Intervention', 'hasNCT2', 'NCT'): (rawGraph['Intervention'], rawGraph['NCTI']),
        ('Outcome', 'hasNCT3', 'NCT'): (rawGraph['Outcome'], rawGraph['NCTO']),
    })

    if isRandomFeature:
        num = dl.get_num()
        hetero_graph.nodes['Problem'].data['feature'] = torch.randn(num['num_P'], 768)
        hetero_graph.nodes['NCT'].data['feature'] = torch.randn(num['num_NCT'], 768)
        hetero_graph.nodes['Intervention'].data['feature'] = torch.randn(num['num_I'], 768)
        hetero_graph.nodes['Outcome'].data['feature'] = torch.randn(num['num_O'], 768)
    else:
        feature = dl.loadAttributeEmb('./AttributeEmbedding.pt')
        hetero_graph.nodes['Problem'].data['feature'] = feature['ProblemFeature']
        hetero_graph.nodes['NCT'].data['feature'] = feature['NCTFeature']
        hetero_graph.nodes['Intervention'].data['feature'] = feature['InterventionFeature']
        hetero_graph.nodes['Outcome'].data['feature'] = feature['OutcomeFeature']
    
    return hetero_graph

def construct_NCT_DGLGraph(dl, similarity, isRandomFeature):
    trailGraph = dl.generateTrailGraph(similarity)
    graph = dgl.graph((trailGraph['sourceNCT'], trailGraph['targetNCT']))
    graph = dgl.add_self_loop(graph)
    
    if isRandomFeature:
        num = dl.get_num()
        graph.ndata['feature'] = torch.randn(num['num_NCT'], 768)
    else:
        feature = dl.loadAttributeEmb('./AttributeEmbedding.pt')
        graph.ndata['feature'] = feature['NCTFeature']
    
    return graph

def compute_loss(pos_score, neg_score):
    n_edges = pos_score.shape[0]
    return (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculateEdgeScore(src, dst):
    return np.array([[sigmoid(src[i].dot(dst[j])) for j in range(len(dst))] for i in range(len(src))])

def construct_negative_graph(graph, k, etype):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,)).to(graph.device)
    return dgl.heterograph({etype: (neg_src, neg_dst)}, num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes}).to(graph.device)

def adjacencyMatrix(src, dst, c, l):
    if len(src) != len(dst):
        raise ValueError('Source and destination lists have different lengths.')
    matrix = np.zeros((c, l), dtype=int)
    for s, d in zip(src, dst):
        matrix[s][d] = 1
    return matrix

def toZeroTrainEdge(dl, matrix):
    train = dl.generateRawTrainGraph()
    for p, nctp in zip(train['Problem'], train['NCTP']):
        matrix[p][nctp] = 0.0
    return matrix

def save_test_set(testProblem, testNCTP, num_P, num_NCT, test_set_file):
    dataset = {
        "testProblem": testProblem,
        "testNCTP": testNCTP,
        "num_P": num_P,
        "num_NCT": num_NCT
    }
    with open(test_set_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

def evaluator(model, dl, best_roc_auc, best_ndcg, best_hr, epoch, loss):
    with torch.no_grad():
        model.eval()
        node_embeddings = model.sage(hetero_graph)
        matrix = calculateEdgeScore(node_embeddings['Problem'].detach().cpu().numpy(), node_embeddings['NCT'].detach().cpu().numpy())
        PreMatrix = toZeroTrainEdge(dl, matrix)
        testSet = dl.generateTestData()
        num = dl.get_num()
        LabelMetrix = adjacencyMatrix(testSet['testProblem'], testSet['testNCTP'], num['num_P'], num['num_NCT'])

        # AUC
        fpr, tpr, _ = roc_curve(LabelMetrix.flatten(), PreMatrix.flatten())
        roc_auc = auc(fpr, tpr)
        print("AUC", roc_auc)

        # NDCG
        ndcg_values = [ndcg_score(PreMatrix, LabelMetrix, k=k) for k in [1, 3, 5, 7, 9, 10, 15, 20]]
        print("NDCG:", ndcg_values)

        # HR
        hr_values = [hits_score(PreMatrix, LabelMetrix, k=k) for k in [1, 3, 5, 7, 9, 10, 15, 20]]
        print("HR:", hr_values)

        # Update best scores
        best_roc_auc = max(best_roc_auc, roc_auc)
        best_ndcg = [max(best_ndcg[i], ndcg_values[i]) for i in range(8)]
        best_hr = [max(best_hr[i], hr_values[i]) for i in range(8)]

        if ndcg_values[7] > 0.22:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            folder_path = f'{folder_name}/{timestamp}'
            os.makedirs(folder_path, exist_ok=True)
            model_path = f'{folder_path}/model.pt'
            torch.save(model.state_dict(), model_path)
            log_path = f'{folder_path}/log.txt'
            with open(log_path, 'a', newline='') as file:
                file.write("NDCG\n")
                file.write(str(ndcg_values))
        
        return best_roc_auc, best_ndcg, best_hr

def trainModel(model, hetero_graph, NCT_graph, epoch, lr, dl):
    k = 5
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_roc_auc = 0
    best_ndcg = [0] * 8
    best_hr = [0] * 8

    with open(filename, 'a', newline='') as file:
        file.write("Best:  Epoch, Loss, ROC_AUC, NDCG, HR, Timestamp\n")
        for i in range(epoch):
            model.train()
            opt.zero_grad()
            negative_graph = construct_negative_graph(hetero_graph, k, ('NCT', 'hasProblem', 'Problem'))
            pos_score, neg_score = model(hetero_graph, NCT_graph, negative_graph, ('NCT', 'hasProblem', 'Problem'))
            loss = compute_loss(pos_score, neg_score)
            loss.backward()
            opt.step()
            print(loss.item())
            best_roc_auc, best_ndcg, best_hr = evaluator(model, dl, best_roc_auc, best_ndcg, best_hr, epoch=i, loss=loss)
        
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        metrics_line = f"{epoch}, {loss.item()}, {best_roc_auc}, {best_ndcg}, {best_hr}, {current_time}\n"
        file.write(metrics_line)

    # 重命名文件
    new_name = filename + str(best_ndcg[7])
    os.rename(filename, new_name)

if __name__ == '__main__':
    # 设置随机数种子
    setup_seed(3407)

    parser = argparse.ArgumentParser(description='Hyperparameters setting')
    parser.add_argument('--isRandomFeature', type=bool, default=False, help='Random feature flag')
    parser.add_argument('--similarity', type=int, default=1, help='Similarity threshold')
    parser.add_argument('--similarity2', type=float, default=0.8, help='Secondary similarity threshold')
    parser.add_argument('--in_feats', type=int, default=768, help='Input features (cannot change)')
    parser.add_argument('--hid_feats', type=int, default=256, help='Hidden features')
    parser.add_argument('--out_feats', type=int, default=128, help='Output features')
    parser.add_argument('--epoch', type=int, default=500, help='Number of epochs')
    parser.add_argument('--n_layers', type=int, default =3, help='Number of layers')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of heads')
    parser.add_argument('--n_fuse_heads', type=int, default=8, help='Number of fuse heads')
    parser.add_argument('--lr', type=float, default=0.002, help='Learning rate')

    args = parser.parse_args()

    print("Received arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    with open(filename, 'w+', newline='') as file:
        for arg, value in vars(args).items():
            file.write(f'{arg} = {value}\n')

    dl = DataLoader()
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    hetero_graph = constructDGLGraph(dl, args.similarity, args.isRandomFeature).to(device)
    NCT_graph = construct_NCT_DGLGraph(dl, args.similarity2, args.isRandomFeature).to(device)

    model = MHAN(hetero_graph, args.in_feats, args.hid_feats, args.out_feats, n_layers=args.n_layers, n_heads=args.n_heads, n_fuse_heads=args.n_fuse_heads)
    model = model.to(device)
    
    trainModel(model, hetero_graph, NCT_graph, args.epoch, args.lr, dl)
