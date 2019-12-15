import random

import pandas as pd
import time
from sklearn.model_selection import train_test_split

import numpy as np


class SVD_plus(object):
    def __init__(self, Klatentvariable=20, lamda=0.1, alpha=0.02, epoch=45, lamda2=0.1, num_batches=100,
                 branch_size=1000):
        self.Klatentvariable = Klatentvariable
        self.lamda = lamda
        self.alpha = alpha
        self.epoch = epoch
        self.lamda2 = lamda2
        self.branch_size = branch_size
        self.num_batches = num_batches

        self.user_num = 0
        self.item_num = 0
        self.R = None
        self.I = None
        self.F_user = None
        self.G_item = None
        self.Bi = None
        self.Bu = None
        self.train_set = None
        self.test_set = None
        self.y = None
        self.train_rmse = []
        self.test_rmse = []

    def fit(self, train, test=None):
        self.mean_v = np.mean(train[:, 2])
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
        self.Bi = np.zeros(self.item_num)
        self.Bu = np.zeros(self.user_num)
        # print("debug:",self.Bi)
        self.y = np.zeros((self.item_num, self.Klatentvariable))
        self.sum_y = 0.1 * np.random.randn(self.user_num, self.Klatentvariable)
        for index in range(len(train)):
            userid = int(train[index][0])
            itemid = int(train[index][1])
            self.R[userid][itemid] = train[index][2]
            self.I[userid][itemid] = 1
        if self.F_user is None:
            self.F_user = 0.1 * np.random.randn(self.user_num, self.Klatentvariable)
        if self.G_item is None:
            self.G_item = 0.1 * np.random.randn(self.item_num, self.Klatentvariable)
        print("Well prepared for SVD++")
        self.Train()

    def Train(self):
        epoch = 0
        rmse_old = 1000
        threshold = 0.000005
        while (epoch < self.epoch):
            epoch += 1
            self.SGD()
            rmse = self.get_rmse()
            t_rmse = self.get_train_rmse()
            print("enpoch:%s, Train_RMSE:%s, Test_RMSE:%s" % (epoch, t_rmse, rmse))
            # if rmse_old - rmse < threshold:
            #     print("rmse_old:", rmse_old)
            #     break
            # else:
            #     rmse_old = rmse

    def SGD(self):
        for u in range(self.user_num):
            # 计算|I|^-0.5
            his_items_num = np.count_nonzero(self.R[u])
            sqrt_num = 0
            if his_items_num > 1:
                sqrt_num = 1 / (his_items_num ** 0.5)
            # 计算sim_y,单独算，为下一步做准备
            self.sum_y = np.zeros((self.user_num, self.Klatentvariable))
            for item in np.nonzero(self.R[u])[0]:
                self.sum_y[u] += self.y[item]
            # 计算Bu,Bi,F,G
            eig_sum = np.zeros(self.Klatentvariable)
            for item in np.nonzero(self.R[u])[0]:
                rating = self.R[u][item]
                predict = self.predict(u, item)
                error = rating - predict
                self.Bu[u] += self.alpha * (error - self.lamda * self.Bu[u])
                self.Bi[item] += self.alpha * (error - self.lamda * self.Bi[item])
                self.F_user[u] += self.alpha * (error * self.G_item[item] - self.lamda2 * self.F_user[u])
                self.G_item[item] += self.alpha * (
                        error * (self.F_user[u] + sqrt_num * self.sum_y[u]) - self.lamda2 * self.G_item[item])
                #     eig_sum += error * sqrt_num * self.G_item[item]
                # # 计算y，单独算，避免影响上一步
                # for item in np.nonzero(self.R[u])[0]:
                y_old = self.y[item]
                self.y[item] += self.alpha * (error * sqrt_num * self.G_item[item] - self.lamda2 * self.y[item])
                self.sum_y[u] += self.y[item] - y_old
        for u in range(self.user_num):
            self.sum_y = np.zeros((self.user_num, self.Klatentvariable))
            for item in np.nonzero(self.R[u])[0]:
                self.sum_y[u] += self.y[item]
        self.alpha *= (0.9 + 0.1 * random.random())
        # print(self.alpha)

    def predict(self, user_id, item_id):
        try:
            his_items_num = np.count_nonzero(self.R[user_id])
        except Exception:
            print(user_id, item_id)
            raise
        sqrt_num = 0
        if his_items_num > 1:
            sqrt_num = 1 / (his_items_num ** 0.5)
        fg = np.dot((self.F_user[user_id] + self.sum_y[user_id] * sqrt_num), self.G_item[item_id])
        score = self.mean_v + self.Bu[user_id] + self.Bi[item_id] + fg
        return score

    def get_rmse(self):
        result = []
        for index in range(len(self.test_set)):
            user = int(self.test_set[index][0])
            item = int(self.test_set[index][1])
            result.append(self.predict(user, item))
        test = [self.test_set[:, 2]]
        rmse = np.sqrt(((np.array(result) - np.array(test)) ** 2).mean())
        return rmse

    def get_train_rmse(self):
        result = []
        for index in range(len(self.train_set)):
            user = int(self.train_set[index][0])
            item = int(self.train_set[index][1])
            result.append(self.predict(user, item))
        test = [self.train_set[:, 2]]
        rmse = np.sqrt(((np.array(result) - np.array(test)) ** 2).mean())
        return rmse

    def standardize(self, num):
        if num > 5.0:
            return 5.0
        elif num < 0.5:
            return 0.5
        tail = num % 1
        # if tail < 0.1:
        #     tail = 0
        # elif tail > 0.9:
        #     tail = 1
        return int(num) + tail


def get_recommendations(trainfilename, predictfilename):
    start_time = time.time()
    train = init(trainfilename)
    target = init(predictfilename, rating=False)
    svd_p = SVD_plus()
    svd_p.fit(train, train)
    result = []
    for index in range(len(target)):
        user = target[index][0]
        item = target[index][1]
        result.append(svd_p.predict(user, item))
    end_time = time.time()
    result = np.array(result)
    printout(result)
    # np.savetxt("optimized_result.csv", result.T, delimiter=",", header="rating")
    print("用时：%s" % (end_time - start_time))
    return


def printout(result):
    f = open("svd_plus2.csv", "w")
    f.write("dataID,rating\n")
    tests = pd.read_csv('test_index.csv', dtype=object, header=None, skiprows=1).values
    count = 0
    for test in tests:
        f.write("%d,%.18f\n" % (count, result[count]))
        count += 1


def get_recommendation(filename, item, user):
    start_time = time.time()
    train = init(filename)
    svd_p = SVD_plus()
    svd_p.fit(train, train)
    print(svd_p.predict(item, user))
    print("RMSE：%s" % svd_p.get_rmse())
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
    svd_p = SVD_plus()
    svd_p.fit(train, test)
    print("RMSE：%s" % svd_p.get_rmse())
    end_time = time.time()
    print("用时：%s" % (end_time - start_time))
    return


if __name__ == '__main__':
    filename = "./train.csv"
    testfilename = "./test_index.csv"
    # get_recommendation(filename, 2345, 468)
    # get_recommendations(filename,testfilename)
    split_and_test(filename)
