import pandas as pd
import time
from sklearn.model_selection import train_test_split

import numpy as np


class PMF(object):
    def __init__(self, Klatentvariable=10, lamda=0.1, epoch=200):
        self.Klatentvariable = Klatentvariable
        self.lamda = lamda
        self.epoch = epoch

        self.user_num = 0
        self.item_num = 0
        self.R = None
        self.I = None
        self.F_user = None
        self.G_item = None
        self.train_set = None
        self.test_set = None

    def set_params(self, parameters):
        if isinstance(parameters, dict):
            self.Klatentvariable = parameters.get("Klatentvariable", 10)
            self.lamda = parameters.get("lamda", 0.1)
            self.epoch = parameters.get("epoch", 200)

    def fit(self, train, test=None):
        # self.mean_v = np.mean(train[:, 2])
        self.train_set = train
        if test is not None:
            self.test_set = test
            self.user_num = int(max(np.amax(train[:, 0]), np.amax(test[:, 0]))) + 1
            self.item_num = int(max(np.amax(train[:, 1]), np.amax(test[:, 1]))) + 1
        else:
            self.user_num = int(np.amax(train[:, 0])) + 1
            self.item_num = int(np.amax(train[:, 1])) + 1
        self.R = np.zeros((self.user_num, self.item_num))
        self.I = np.zeros((self.user_num, self.item_num))
        for index in range(len(train)):
            userid = int(train[index][0])
            itemid = int(train[index][1])
            self.R[userid][itemid] = train[index][2]
            self.I[userid][itemid] = 1
        if self.F_user is None:
            self.F_user = 0.1 * np.random.randn(self.Klatentvariable, self.user_num)
        if self.G_item is None:
            self.G_item = 0.1 * np.random.randn(self.Klatentvariable, self.item_num)
        print("Well prepared for ALS")
        self.ALS()

    def ALS(self):
        G_item_inc = np.zeros((self.Klatentvariable, self.item_num))  # 创建电影 M x D 0矩阵
        F_user_inc = np.zeros((self.Klatentvariable, self.user_num))  # 创建用户 N x D 0矩阵
        enpoch = 0
        lamda = self.lamda
        E = np.identity(self.Klatentvariable)
        while ((not self.almost_same(self.F_user, F_user_inc) or not self.almost_same(self.G_item,
                                                                                      G_item_inc)) and enpoch < self.epoch):
            # print("enpoch now is %s" % enpoch)
            enpoch += 1
            G_item_inc = self.G_item.copy()
            F_user_inc = self.F_user.copy()
            # Fix G_item_inc and estimate F_user_inc
            gTg = [np.dot(np.array([self.G_item[:, m]]).T, np.array([self.G_item[:, m]])) for m in range(self.item_num)]
            for i, Ii in enumerate(self.I):
                nui = np.count_nonzero(Ii)  # Number of items user i has rated
                if nui == 0: nui = 1  # Be aware of zero counts!
                # Least squares solution
                Ai = lamda * nui * E
                Vi = np.zeros((self.Klatentvariable, 1))
                score = self.R[i]
                for index in np.nonzero(Ii)[0]:
                    Ai += gTg[index]
                    Vi += np.array([self.G_item[:, index] * score[index]]).T
                self.F_user[:, i] = np.linalg.solve(Ai, Vi).T
            # print("F %s" % enpoch)
            # Fix F_user_inc and estimate G_item_inc
            fTf = [np.dot(np.array([self.F_user[:, m]]), np.array([self.F_user[:, m]]).T) for m in range(self.user_num)]
            for j, Ij in enumerate(self.I.T):
                nmj = np.count_nonzero(Ij)  # Number of users that rated item j
                if (nmj == 0): nmj = 1  # Be aware of zero counts!
                # Least squares solution
                Aj = lamda * nmj * E
                Vj = np.zeros((self.Klatentvariable, 1))
                score = self.R[:, j]
                for index in np.nonzero(Ij)[0]:
                    Aj += fTf[index]
                    Vj += np.array([self.F_user[:, index] * score[index]]).T
                self.G_item[:, j] = np.linalg.solve(Aj, Vj).T
            # print("G %s" % enpoch)
        print("ALS finished, enpoch now is %s" % enpoch)

    def ALS_old(self):
        G_item_inc = np.zeros((self.Klatentvariable, self.item_num))  # 创建电影 M x D 0矩阵
        F_user_inc = np.zeros((self.Klatentvariable, self.user_num))  # 创建用户 N x D 0矩阵
        enpoch = 0
        lamda = self.lamda
        E = np.identity(self.Klatentvariable)
        while ((not self.almost_same(self.F_user, F_user_inc) or not self.almost_same(self.G_item,
                                                                                      G_item_inc)) and enpoch < self.epoch):
            # print("enpoch now is %s" % enpoch)
            enpoch += 1
            G_item_inc = self.G_item.copy()
            F_user_inc = self.F_user.copy()
            # Fix G_item_inc and estimate F_user_inc
            for i, Ii in enumerate(self.I):
                nui = np.count_nonzero(Ii)  # Number of items user i has rated
                if nui == 0: nui = 1  # Be aware of zero counts!

                # Least squares solution
                Ai = np.dot(self.G_item, np.dot(np.diag(Ii), self.G_item.T)) + lamda * nui / 10 * E
                Vi = np.dot(self.G_item, np.dot(np.diag(Ii), self.R[i].T))
                self.F_user[:, i] = np.linalg.solve(Ai, Vi)
            print("F %s" % enpoch)
            # Fix F_user_inc and estimate G_item_inc
            for j, Ij in enumerate(self.I.T):
                nmj = np.count_nonzero(Ij)  # Number of users that rated item j
                if (nmj == 0): nmj = 1  # Be aware of zero counts!

                # Least squares solution
                Aj = np.dot(self.F_user, np.dot(np.diag(Ij), self.F_user.T)) + lamda * nmj / 10 * E
                Vj = np.dot(self.F_user, np.dot(np.diag(Ij), self.R[:, j]))
                self.G_item[:, j] = np.linalg.solve(Aj, Vj)
            print("G %s" % enpoch)
        print("ALS finished, enpoch now is %s" % enpoch)

    def almost_same(self, new, old):
        diff = np.linalg.norm(new - old)
        # print(diff)
        threshold = 2
        if diff <= threshold:
            print("almost same %s" % diff)
            return True
        else:
            return False

    def predict(self, user_id, item_id):
        return np.dot(self.F_user[:, int(user_id)], self.G_item[:, int(item_id)])

    def get_cmse(self):
        result = []
        for index in range(len(self.test_set)):
            user = self.test_set[index][0]
            item = self.test_set[index][1]
            result.append(self.predict(user, item))
        test = [self.test_set[:, 2]]
        rmse = np.sqrt(((np.array(result) - np.array(test)) ** 2).mean())
        return rmse


def get_recommendations(trainfilename, predictfilename):
    start_time = time.time()
    train = init(trainfilename)
    target = init(predictfilename, rating=False)
    pmf = PMF()
    pmf.set_params({"Klatentvariable": 10, "lamda": 0.5})
    pmf.fit(train)
    result = []
    for index in range(len(target)):
        user = target[index][0]
        item = target[index][1]
        result.append(pmf.predict(user, item))
    end_time = time.time()
    result = np.array(result)
    np.savetxt("modelbase_result.csv", result.T, delimiter=",", header="rating")
    print("用时：%s" % (end_time - start_time))
    return


def get_recommendation(filename, item, user):
    start_time = time.time()
    train, test = split(filename)
    pmf = PMF()
    pmf.set_params({"Klatentvariable": 10, "lamda": 0.1})
    pmf.fit(train, test)
    print(pmf.predict(item, user))
    end_time = time.time()
    print("用时：%s" % (end_time - start_time))
    return


def init(filename, rating=True):
    data = []
    file = open(filename, "r", encoding="UTF-8")
    for line in file.readlines()[1:]:  # 打开指定文件
        if rating:
            (userid, itemid, rating, ts) = line.split(',')  # 数据集中每行有4项
            uid = int(userid)
            mid = int(itemid)
            rat = float(rating)
            data.append([uid, mid, rat])
        else:
            (userid, itemid) = line.split(',')  # 数据集中每行有4项
            uid = int(userid)
            mid = int(itemid)
            data.append([uid, mid])
    print('Data preparation finished')
    return np.array(data)


def split(file):
    csv = init(file)
    train, test = train_test_split(csv, test_size=0.2)
    return train, test


def split_and_test(filename):
    start_time = time.time()
    train, test = split(filename)
    pmf = PMF()
    pmf.set_params({"Klatentvariable": 10, "lamda": 0.1, "epoch": 200})
    pmf.fit(train, test)
    print("RMSE：%s" % pmf.get_cmse())
    end_time = time.time()
    print("用时：%s" % (end_time - start_time))
    return


if __name__ == '__main__':
    filename = "./train.csv"
    testfilename = "./test_index.csv"
    get_recommendation(filename, 2345, 468)
    # split_and_test(filename)
    # get_recommendations(filename, testfilename)
