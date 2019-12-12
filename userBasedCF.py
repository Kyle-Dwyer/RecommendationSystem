import time
from operator import itemgetter
import numpy as np
import pandas as pd


def get_recommendations(trainfilename, predictfilename):
    start_time = time.time()
    train = pd.read_csv(trainfilename)
    target = pd.read_csv(predictfilename)
    data = init(train)
    A, C, W = user_similarity(data)
    user_item_list = np.array(target)
    result = []
    for index in range(len(user_item_list)):
        user = user_item_list[index][0]
        item = user_item_list[index][1]
        result.append(A[user] + predict(user, item, data, W, 5))
    end_time = time.time()
    result = np.array(result)
    np.savetxt("userbase_result.csv", result.T, delimiter=",", header="rating")
    print("用时：%s" % (end_time - start_time))
    return


def get_recommendation(filename, user, item, K=5):
    start_time = time.time()
    csv = pd.read_csv(filename)
    # file = open(filename, "r", encoding="UTF-8")
    data = init(csv)
    A, C, W = user_similarity(data)
    print(A[user] + predict(user, item, data, W, K))
    end_time = time.time()
    print("用时：%s" % (end_time - start_time))
    return


def init(file):
    data = {}
    # for line in file.readlines()[1:]:
    #     line = line.strip().split(',')
    #     if not line[0] in data.keys():
    #         data[line[0]] = {line[1]: float(line[2])}
    #     else:
    #         data[line[0]][line[1]] = float(line[2])
    for user, others in file.groupby('userID'):
        item_list = others['itemID'].tolist()
        score_list = others['rating'].tolist()
        data[user] = dict(zip(item_list, score_list))
    print('Data preparation finished')
    return data


def split_and_test(filename, K=5):
    csv = pd.read_csv(filename)
    train, test = train_test_split(csv, 2)
    A, C, W = user_similarity(train)
    result = {}
    for u, items in test.items():
        avg = A[u]
        result[u] = {}
        for item in items.keys():
            result[u][item] = avg + predict(u, item, train, W, K)
    predictions = [result[u][v] for u in result.keys() for v in result[u].keys()]
    targets = [test[u][v] for u in test.keys() for v in test[u].keys()]
    rmse = get_rmse(np.array(predictions), np.array(targets))
    print("RMSE:%s" % rmse)


def train_bestK(filename):
    csv = pd.read_csv(filename)
    train, test = train_test_split(csv, 2)
    A, C, W = user_similarity(train)
    result = {}
    K = 3
    for u, items in test.items():
        avg = A[u]
        result[u] = {}
        for item in items.keys():
            result[u][item] = avg + predict(u, item, train, W, K)
    predictions = [result[u][v] for u in result.keys() for v in result[u].keys()]
    targets = [test[u][v] for u in test.keys() for v in test[u].keys()]
    rmse = get_rmse(np.array(predictions), np.array(targets))
    for j in range(4, 100):
        print(j)
        for u, items in test.items():
            avg = A[u]
            result[u] = {}
            for item in items.keys():
                result[u][item] = avg + predict(u, item, train, W, j)
        predictions = [result[u][v] for u in result.keys() for v in result[u].keys()]
        targets = [test[u][v] for u in test.keys() for v in test[u].keys()]
        new_rmse = get_rmse(np.array(predictions), np.array(targets))
        print(new_rmse)
        if new_rmse < rmse:
            K = j
            rmse = new_rmse
    print("RMSE:%s" % rmse)
    return K


def get_rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def train_test_split(data, seed):
    train_set = {}
    test_set = {}
    for user, movies in data.groupby('userID'):
        movies = movies.sample(
            frac=1, random_state=None).reset_index(drop=True)
        train = movies[:int(0.8 * len(movies))]
        test = movies[int(0.8 * len(movies)):]
        item_list = train['itemID'].tolist()
        score_list = train['rating'].tolist()
        item_list2 = test['itemID'].tolist()
        score_list2 = test['rating'].tolist()
        train_set[user] = dict(zip(item_list, score_list))
        test_set[user] = dict(zip(item_list2, score_list2))
    print('Data preparation finished')
    return train_set, test_set


def user_similarity(trainset):
    # trainset = {'u1': {'i1': 1, 'i2': 3}, 'u2': {'i1': 1, 'i2': 2, 'i3': 3}, 'u3': {'i1': 3, 'i3': 3}}
    # print(trainset)
    # print("----------------")
    # 初始化用户关系
    C = {}  # 用户联系集
    N = {}  # 用户关联的item总数
    W = {}  # 相似度
    for u in trainset.keys():
        C[u] = {}
        W[u] = {}
        for v in trainset.keys():
            if v == u:
                continue
            C[u][v] = set()
            W[u][v] = 0
    # 平均值
    A = {}
    for u in trainset.keys():
        N[u] = len(trainset[u])
        totle = sum(score for item, score in trainset[u].items())
        A[u] = totle / N[u]
    # print(A)
    # 标准化
    for u in trainset.keys():
        for v in trainset[u].keys():
            trainset[u][v] -= A[u]
    # print(trainset)
    # 计算产品集,稀疏矩阵大大减少运算量
    item_user = {}
    for u, items in trainset.items():
        for i, j in items.items():
            if i not in item_user.keys():
                item_user[i] = {u: int(j)}
            item_user[i][u] = int(j)
    print('Inverse table finished')
    # print(item_user)
    # print("----------------")
    # 获取公共item
    for i, users in item_user.items():
        for u in users:
            for v in users:
                if v == u:
                    continue
                C[u][v].add(i)
    print('Co-rated items count finished')
    # Calculate similarity matrix
    for u, related_users in C.items():
        for v, items in related_users.items():
            temp1 = sum((trainset[u][item] * trainset[v][item]) for item in items)
            temp2 = sum(trainset[u][item] ** 2 for item in items) ** 0.5
            temp3 = sum(trainset[v][item] ** 2 for item in items) ** 0.5
            if temp2 + temp3 == 0:
                W[u][v] = 1.0
            elif temp2 == 0:
                W[u][v] = sum(trainset[v][item] for item in items) / (len(items) ** 0.5 * temp3)
            elif temp3 == 0:
                W[u][v] = sum(trainset[u][item] for item in items) / (len(items) ** 0.5 * temp2)
            else:
                W[u][v] = temp1 / (temp2 * temp3)
    print('Similarity calculation finished')
    # print(N)
    # print(C)
    # print(W)
    return A, C, W


def predict(user, item, train, W, K):
    already_items = train[user]
    if item in already_items.keys():
        return already_items[item]
    molecular = 0
    denominator = 0
    for v, wuv in sorted(W[user].items(), key=itemgetter(1), reverse=True)[:K]:
        score = train[v].get(item, -1)
        if score == -1:
            continue
        else:
            molecular += W[user][v] * score
            denominator += abs(W[user][v])
    predict_score = 0 if denominator == 0 else molecular / denominator
    return predict_score


if __name__ == '__main__':
    filename = "./train.csv"
    testfilename = "./test_index.csv"
    # get_recommendation(filename, 722, 405,5)
    get_recommendations(filename, testfilename)
    split_and_test(filename)
    # Ks = []
    # for i in range(10):
    #     Ks.append(train_bestK(filename))
    #     print(Ks)
    #     print(np.mean(Ks))

    # print(A["2345"] + predict("2345", "468", data, W, 5))
    # 2.9655172413793105

# Data preparation finished
# Inverse table finished
# Co-rated items count finished
# Similarity calculation finished
# RMSE:1.023412024277581
# [3]
# 3.0
# Data preparation finished
# Inverse table finished
# Co-rated items count finished
# Similarity calculation finished
# RMSE:1.0167643274161744
# [3, 3]
# 3.0
# Data preparation finished
# Inverse table finished
# Co-rated items count finished
# Similarity calculation finished
# RMSE:1.020179181639595
# [3, 3, 3]
# 3.0
# Data preparation finished
# Inverse table finished
# Co-rated items count finished
# Similarity calculation finished
# RMSE:1.0323924873473138
# [3, 3, 3, 3]
# 3.0
# Data preparation finished
# Inverse table finished
# Co-rated items count finished
# Similarity calculation finished
# RMSE:1.028388123812888
# [3, 3, 3, 3, 3]
# 3.0
# Data preparation finished
# Inverse table finished
# Co-rated items count finished
# Similarity calculation finished
# RMSE:1.0260342102196678
# [3, 3, 3, 3, 3, 3]
# 3.0
# Data preparation finished
# Inverse table finished
# Co-rated items count finished
# Similarity calculation finished
# RMSE:1.0203176431075807
# [3, 3, 3, 3, 3, 3, 3]
# 3.0
# Data preparation finished
# Inverse table finished
# Co-rated items count finished
# Similarity calculation finished
# RMSE:1.0262436064949274
# [3, 3, 3, 3, 3, 3, 3, 3]
# 3.0
# Data preparation finished
# Inverse table finished
# Co-rated items count finished
# Similarity calculation finished
# RMSE:1.0331355714394508
# [3, 3, 3, 3, 3, 3, 3, 3, 3]
# 3.0
# Data preparation finished
# Inverse table finished
# Co-rated items count finished
# Similarity calculation finished
# RMSE:1.023695809453493
# [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
# 3.0
