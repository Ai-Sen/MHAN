import csv
from imghdr import tests
import sys
import torch
import torch.nn.functional as F
import numpy as np
from random import shuffle


class DataLoader:
    
    def __init__(self, num_P=100, num_NCT=2340, num_I=3675, num_O=4032):
        
        """
        记录每一类节点的个数
        MATCH (i:Intervention)
        WITH COUNT(i) AS NumberOfInterventions
        MATCH (n:NCT)
        WITH NumberOfInterventions, COUNT(n) AS NumberOfNCTs
        MATCH (o:Outcome)
        WITH NumberOfInterventions, NumberOfNCTs, COUNT(o) AS NumberOfOutcomes
        MATCH (p:Problem)
        RETURN NumberOfInterventions, NumberOfNCTs, NumberOfOutcomes, COUNT(p) AS NumberOfProblems
        """
        self.num_P = num_P
        self.num_NCT = num_NCT
        self.num_I = num_I
        self.num_O = num_O
        # name official_title  title condition
        # Intervention NCT  Outcome Problem
        self.indexI=0
        self.indexN=self.num_I
        
        self.indexO=self.indexN+self.num_NCT
        self.indexP=self.indexO+self.num_O

        self.NCTP, self.NCTI, self.NCTO, self.Problem, self.Intervention, self.Outcome = self._loadRawData_('./data/ClinicalTrails.csv')
        self.testNCTP, self.testProblem, self.trainNCTP, self.trainProblem = self._dataPartition_(self.NCTP, self.Problem)

        
    def get_num(self):
        return {'num_P': self.num_P,
                'num_NCT': self.num_NCT,
                'num_I': self.num_I,
                'num_O': self.num_O}


    def _loadRawData_(self, path):
        '''
            :param path: 原始文件路径，第1列为NCT，第2列是关系，第3列是P, I, O三种点
                         从P开始从0编号，然后是NCT然后I然后O
            :return: 元组，每个元素为List: NCTP, NCTI, NCTO, Problem, Intervention, Outcome. 
                     第i个与第i+3个点及他们的关系可对应构成三元组
        '''
        with open(path) as f:
            reader = csv.reader(f)
            data = list(reader)
        NCTP = []
        NCTI = []
        NCTO = []
        Problem = []
        Intervention = []
        Outcome = []
        
        
        for triad in data:
            if triad[1] == 'hasProblem':
                NCTP.append(int(triad[0]) - self.indexN)
                Problem.append(int(triad[2]) - self.indexP)
            elif triad[1] == 'hasIntervention':
                NCTI.append(int(triad[0]) - self.indexN)
                Intervention.append(int(triad[2]) - self.indexI)
            elif triad[1] == 'hasOutcome':
                NCTO.append(int(triad[0]) - self.indexN)
                Outcome.append(int(triad[2]) - self.indexO)
            else:
                sys.exit('Invalid meta path: ', triad)
        # print(type(NCTP[5]))
        # print(Problem)
        f.close()
        return NCTP, NCTI, NCTO, Problem, Intervention, Outcome


    def _dataPartition_(self, NCTP, Problem):
        '''
            进行user->item训练集、测试集的划分
        '''
        testSize = int(len(NCTP)*0.2)# 测试集比例
        num_Problem = len(Problem)
        testProblem = list()
        testNCTP = list()
        trainProblem = list()
        trainNCTP = list()
        pairs = list(zip(Problem, NCTP))
        shuffle(pairs)
        # print(pairs)
        num_test = 0
        num = 0
        while num < num_Problem:
            # 这里不能把点删了，有些点只出现过一次，这条路放到测试集中，点就没了
            if num_test <= testSize and Problem.count(pairs[num][0]) != 1 and NCTP.count(pairs[num][1]) != 1:
                testProblem.append(pairs[num][0])
                testNCTP.append(pairs[num][1])
                num_test = num_test+1
                num = num+1
            else:
                Problem.remove(pairs[num][0])
                NCTP.remove(pairs[num][1])
                trainProblem.append(pairs[num][0])
                trainNCTP.append(pairs[num][1])
                # testProblem.append(pairs[num][0])
                # testNCTP.append(pairs[num][1])
                num = num+1
                
        
        # step = len(NCTP) // (len(NCTP) * 0.2)
        # for i in range(len(NCTP)):
        #     # 这里不能把点删了，有些点只出现过一次，这条路放到测试集中，点就没了
        #     if i % step == 0 and Problem.count(Problem[i]) != 1 and NCTP.count(NCTP[i]) != 1:
        #         testNCTP.append(NCTP[i])
        #         testProblem.append(Problem[i])
        #     else:
        #         trainNCTP.append(NCTP[i])
        #         trainProblem.append(Problem[i])
        #         testNCTP.append(NCTP[i])
        #         testProblem.append(Problem[i])

        # print(len(testNCTP), len(trainNCTP))
        # print(len(testProblem), len(trainProblem))
        return testNCTP, testProblem, trainNCTP, trainProblem


    def generateRawTrainGraph(self):
        return {'NCTP': self.trainNCTP, 
                'NCTI': self.NCTI, 
                'NCTO': self.NCTO, 
                'Problem': self.trainProblem, 
                'Intervention': self.Intervention, 
                'Outcome': self.Outcome}


    def generateTestData(self):
        return {'testProblem': self.testProblem, 
                'testNCTP': self.testNCTP}


    def _similarNCT_(self, path, similarity):
        '''
            计算出相似度高的NCT节点，两个相似的NCT是双向有向边
        '''
        head = []
        tail = []
        distanceList = []
        # 加载NCT的attribute embedding
        attributeEmbedding = torch.load(path)
        NCTAttributeEmb = attributeEmbedding[self.num_P:self.num_P+self.num_NCT]
        # 计算NCT两两之间的L2范数，即距离，有从a->b, 也有从b->a
        for i in range(len(NCTAttributeEmb)):
            for j in range(len(NCTAttributeEmb)):
                distance = F.pairwise_distance(NCTAttributeEmb[i], NCTAttributeEmb[j], p=2).tolist()  # type为float, 不是list, 因为只有一个
                if i != j:
                    head.append(i)
                    tail.append(j)
                    distanceList.append(distance)
        # 将距离归一化
        distanceList = np.array(distanceList)
        similarityList = 1 - (distanceList - np.min(distanceList)) / (np.max(distanceList) - np.min(distanceList))
        similarityList = similarityList.tolist()
        # 挑选相似度
        finalHead = []
        finalTail = []
        counter = 0
        for i in range(len(NCTAttributeEmb)):
            finalHead.append(i)
            finalTail.append(i)
        for i in range(len(similarityList)):
            if similarityList[i] > similarity:
                finalHead.append(head[i])
                finalTail.append(tail[i])
                counter = counter+1
        print('related edge:')
        print(counter)
        return finalHead, finalTail, counter


    def generateTrailGraph(self, similarity):
        head, tail, counter = self._similarNCT_('./data/AttributeEmbedding.pt', similarity)
        return {'sourceNCT': head,
                'targetNCT': tail}


    def loadAttributeEmb(self, path):
        loadTensor = torch.load(path)
        ProblemFeature = loadTensor[self.indexP : self.indexP+self.num_P]
        NCTFeature = loadTensor[self.indexN : self.indexN+self.num_NCT]
        InterventionFeature = loadTensor[self.indexI : self.indexI+self.num_I]
        OutcomeFeature = loadTensor[self.indexO:self.indexO+self.num_O]
        # print(ProblemFeature.shape)
        # print(ProblemFeature)
        # print(OutcomeFeature.shape)
        # print(OutcomeFeature)
        return {'ProblemFeature': ProblemFeature, 
                'NCTFeature': NCTFeature, 
                'InterventionFeature': InterventionFeature, 
                'OutcomeFeature': OutcomeFeature}


if __name__ == '__main__':
    dl = DataLoader()
    final = dl.generateTrailGraph(1.0)
    print(final['sourceNCT'])
    # print(train['NCTP'])
    # print(train['Problem'])
    