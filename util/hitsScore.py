import numpy as np

def hits_score(pre, test, k):
    # 逆序,默认的输出每行元素的索引值。这些索引值对应的元素是从小到大排序的。
    pre = np.argsort(-pre)
    # 计算出测试集中每个Problem推荐出的个数
    NumberOfHits = 0
    GT = test.sum()
    for item in range(len(test)):
        if sum(test[item]) != 0:
            rel = test[item][pre[item]]
            NumberOfHits = NumberOfHits + rel[:k].sum()
    return NumberOfHits / GT


# def hits_score(pre, test, k):
#     # 逆序,默认的输出每行元素的索引值。这些索引值对应的元素是从小到大排序的。
#     pre = np.argsort(-pre)
#     # 计算出测试集中每个Problem推荐出的个数
#     test_num = test.sum(axis=1)
#     NumberOfHits = 0
#     GT = 0
#     for item in range(len(test)):
#         if sum(test[item]) != 0:
#             rel = test[item][pre[item]]
#             # print(rel)
#             # print(rel[:k].sum())
#             NumberOfHits = NumberOfHits + rel[:k].sum()
#             GT = GT + test_num[item]
#     return NumberOfHits / GT


if __name__ == "__main__":
    test = np.array([[1, 0, 0, 1], [1, 1, 0, 0], [0, 1, 0, 0]])
    pre = np.array([[4, 2, 3, 55], [5, 6, 37, 8], [-7, 68, 9, 0]])
    print(hits_score(pre, test, 1))
