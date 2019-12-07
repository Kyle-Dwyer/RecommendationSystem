import time
from operator import itemgetter
import numpy as np
import pandas as pd


def get_recommendations(trainfilename, predictfilename):
    start_time = time.time()
    train = pd.read_csv(trainfilename)
    target = pd.read_csv(predictfilename)
    data = init(train)
    C, W = item_similarity(data)
    user_item_list = np.array(target)
    result = []
    for index in range(len(user_item_list)):
        user = user_item_list[index][0]
        item = user_item_list[index][1]
        result.append(predict(item, user, data, W, 200))
    end_time = time.time()
    result = np.array(result)
    np.savetxt("itembase_result.csv", result.T, delimiter=",", header="rating")
    print("用时：%s" % (end_time - start_time))
    return


def get_recommendation(filename, item, user, K=5):
    start_time = time.time()
    csv = pd.read_csv(filename)
    # file = open(filename, "r", encoding="UTF-8")
    data = init(csv)
    C, W = item_similarity(data)
    print(predict(item, user, data, W, K))
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
    for item, others in file.groupby('itemID'):
        user_list = others['userID'].tolist()
        score_list = others['rating'].tolist()
        data[item] = dict(zip(user_list, score_list))
    print('Data preparation finished')
    return data


def split_and_test(filename, K=5):
    csv = pd.read_csv(filename)
    train, test = train_test_split(csv, 2)
    C, W = item_similarity(train)
    result = {}
    for item, users in test.items():
        result[item] = {}
        for user in users.keys():
            result[item][user] = predict(item, user, train, W, K)
    predictions = [result[u][v] for u in result.keys() for v in result[u].keys()]
    targets = [test[u][v] for u in test.keys() for v in test[u].keys()]
    rmse = get_rmse(np.array(predictions), np.array(targets))
    print("RMSE:%s" % rmse)


def train_bestK(filename):
    csv = pd.read_csv(filename)
    train, test = train_test_split(csv, 2)
    C, W = item_similarity(train)
    result = {}
    K = 390
    for item, users in test.items():
        result[item] = {}
        for user in users.keys():
            result[item][user] = predict(item, user, train, W, K)
    predictions = [result[u][v] for u in result.keys() for v in result[u].keys()]
    targets = [test[u][v] for u in test.keys() for v in test[u].keys()]
    rmse = get_rmse(np.array(predictions), np.array(targets))
    print("RMSE:%s" % rmse)
    for j in range(400, 1000, 10):
        print(j)
        for item, users in test.items():
            result[item] = {}
            for user in users.keys():
                result[item][user] = predict(item, user, train, W, j)
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
    for item, movies in data.groupby('itemID'):
        movies = movies.sample(
            frac=1, random_state=None).reset_index(drop=True)
        train = movies[:int(0.8 * len(movies))]
        test = movies[int(0.8 * len(movies)):]
        user_list = train['userID'].tolist()
        score_list = train['rating'].tolist()
        user_list2 = test['userID'].tolist()
        score_list2 = test['rating'].tolist()
        train_set[item] = dict(zip(user_list, score_list))
        test_set[item] = dict(zip(user_list2, score_list2))
    print('Data preparation finished')
    return train_set, test_set


def item_similarity(trainset):
    item_user = trainset
    user_item = {}
    for item, users in trainset.items():
        for i, j in users.items():
            if i not in user_item.keys():
                user_item[i] = {item: int(j)}
                user_item[i][item] = int(j)
    print('Inverse table finished')
    # 初始化item关系
    C = {}  # item联系集
    N = {}  # item关联的user总数
    W = {}  # 相似度
    for i in item_user.keys():
        C[i] = {}
        W[i] = {}
        for j in item_user.keys():
            if j == i:
                continue
            C[i][j] = set()
            W[i][j] = 0
    # 获取公共user
    for user, items in user_item.items():
        for i in items:
            for j in items:
                if j == i:
                    continue
                C[i][j].add(user)
    print('Co-rated items count finished')
    # Calculate similarity matrix
    for i, related_items in C.items():
        for j, users in related_items.items():
            temp1 = sum((user_item[user][i] * user_item[user][j]) for user in users)
            temp2 = sum(user_item[user][i] ** 2 for user in users) ** 0.5
            temp3 = sum(user_item[user][j] ** 2 for user in users) ** 0.5
            if temp2 + temp3 == 0:
                W[i][j] = 1.0
            elif temp2 == 0:
                W[i][j] = sum(user_item[user][j] for user in users) / (len(users) ** 0.5 * temp3)
            elif temp3 == 0:
                W[i][j] = sum(user_item[user][i] for user in users) / (len(users) ** 0.5 * temp2)
            else:
                W[i][j] = temp1 / (temp2 * temp3)
    print('Similarity calculation finished')
    # print(N)
    # print(C)
    # print(W)
    return C, W


def predict(item, user, train, W, K):
    already_items = train[item]
    if user in already_items.keys():
        return already_items[user]
    molecular = 0
    denominator = 0
    for v, wuv in sorted(W[item].items(), key=itemgetter(1), reverse=True)[:K]:
        score = train[v].get(user, -1)
        if score == -1:
            continue
        else:
            molecular += W[item][v] * score
            denominator += abs(W[item][v])
    predict_score = 0 if denominator == 0 else molecular / denominator
    return predict_score


if __name__ == '__main__':
    filename = "./train.csv"
    testfilename = "./test_index.csv"
    # get_recommendation(filename, 468, 2356, 10)
    # split_and_test(filename, 20)
    get_recommendations(filename, testfilename)
    # Ks = []
    # for i in range(1):
    #     Ks.append(train_bestK(filename))
    #     print(Ks)
    #     print(np.mean(Ks))

    # print(A["2345"] + predict("2345", "468", data, W, 5))
    # 2.9655172413793105
