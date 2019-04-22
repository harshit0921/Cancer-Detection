# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 03:14:54 2019

@author: Harshit Gupta
"""

import numpy as np

class SVM:
    # init the structure with parameters
    def __init__(self, x_train, y_train, C, tolerance):
        self.Xtrain = np.mat(x_train)
        self.ytrain = np.mat(y_train).T
        self.C = C
        self.tol = tolerance
        self.m = self.Xtrain.shape[0]
        self.n = self.Xtrain.shape[1]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.w = np.mat(np.zeros(self.n)).T
        self.e_cache = np.mat(np.zeros(self.m)).T
        self.K = np.matmul(self.Xtrain, self.Xtrain.T)
    
    
    def take_step(self, i1, i2):
        alpha1 = self.alphas[i1]
        y1 = self.ytrain[i1]
        if alpha1 > 0 and alpha1 < self.C:
            E1 = self.e_cache[i1]
        else:
            E1 = self.Xtrain[i1] * self.w + self.b - self.ytrain[i1]
        alpha2 = self.alphas[i2]
        y2 = self.ytrain[i2]
        E2 = self.e_cache[i2]
        s = y1 * y2
        if y1 == y2:
            L = max(0, alpha1+alpha2-self.C)
            H = min(self.C, alpha1+alpha2)
        else:
            L = max(0, alpha2-alpha1)
            H = min(self.C, self.C+alpha2-alpha1)
        if L == H:
            return 0
        eta = self.K[i1, i1] + self.K[i2, i2] - 2*self.K[i1, i2]
        if eta > 0:
            a2 = alpha2 + y2*(E1-E2)/eta
            if a2 < L:
                a2 = L
            elif a2 > H:
                a2 = H
        else:
            c1 = eta / 2.0
            c2 = y2 * (E1 - E2) - eta * alpha2
            Lobj = c1 * L * L + c2 * L
            Hobj = c1 * H * H + c2 * H
            if Lobj > Hobj + self.tol:
                a2 = L
            elif Lobj < Hobj - self.tol:
                a2 = H
            else:
                a2 = alpha2
        if abs(a2 - alpha2) < self.tol:
            return 0
        a1 = alpha1 - s*(a2 - alpha2)
        if a1 > 0 and a1 < self.C:
            bnew = self.b - E1 - y1 * (a1 - alpha1) * self.K[i1, i1] - y2 * (a2 - alpha2) * self.K[i1, i2]
        elif a2 > 0 and a2 < self.C:
            bnew = self.b - E2 - y1 * (a1 - alpha1) * self.K[i1, i2] - y2 * (a2 - alpha2) * self.K[i2, i2]
        else:
            b1 = self.b - E1 - y1 * (a1 - alpha1) * self.K[i1, i1] - y2 * (a2 - alpha2) * self.K[i1, i2]
            b2 = self.b - E2 - y1 * (a1 - alpha1) * self.K[i1, i2] - y2 * (a2 - alpha2) * self.K[i2, i2]
            bnew = (b1 + b2) / 2.0
        self.b = bnew
        self.alphas[i1] = a1
        self.alphas[i2] = a2
        self.w = self.Xtrain.T * np.multiply(self.alphas, self.ytrain)
        for i in range(self.m):
            if (self.alphas[i] > 0) and (self.alphas[i] < self.C):
                self.e_cache[i] = self.Xtrain[i] * self.w + self.b - self.ytrain[i]
        return 1
    
    def examine_example(self, i2):
        y2 = self.ytrain[i2]
        alpha2 = self.alphas[i2]
        if alpha2 > 0 and alpha2 <self.C:
            E2 = self.e_cache[i2]
        else:
            E2 = self.Xtrain[i2] * self.w + self.b - self.ytrain[i2]
            self.e_cache[i2] = E2
        r2 = E2 * y2
        if((r2 < -self.tol) and (self.alphas[i2] < self.C)) or ((r2 > self.tol) and (self.alphas[i2] > 0)):
            # heuristic 1: find the max deltaE
            max_delta_E = 0
            i1 = -1
            for i in range(self.m):
                if self.alphas[i] > 0 and self.alphas[i] < self.C:
                    if i == i2:
                        continue
                    E1 = self.e_cache[i]
                    delta_E = abs(E1 - E2)
                    if delta_E > max_delta_E:
                        max_delta_E = delta_E
                        i1 = i
            if i1 >= 0:
                if self.take_step(i1, i2):
                    return 1
            # heuristic 2: find the suitable i1 on border at random
            random_index = np.random.permutation(self.m)
            for i in random_index:
                if self.alphas[i] > 0 and self.alphas[i] < self.C:
                    if i == i2:
                        continue
                    if self.take_step(i, i2):
                        return 1
            # heuristic 3: find the suitable i1 at random on all alphas
            random_index = np.random.permutation(self.m)
            for i in random_index:
                if i == i2:
                    continue
                if self.take_step(i1, i2):
                    return 1
        return 0
    
    def main_routine(self, max_iter=10):
        num_changed = 0
        examine_all = 1
        passes = 0
        while(passes < max_iter):
            num_changed = 0
            if (examine_all == 1):
                # loop over all training examples
                for i in range(self.m):
                    num_changed += self.examine_example(i)
            else:
                for i in range(self.m):
                    if (self.alphas[i] > 0) and (self.alphas[i] < self.C):
                        num_changed += self.examine_example(i)
            if (num_changed == 0):
                passes += 1
            if (examine_all == 1):
                examine_all = 0
            elif (num_changed == 0):
                examine_all = 1
            
            print("Num Changed: {}, Pass: {}, Examine All: {}".format(num_changed, passes, examine_all))
            if(num_changed == 0 and examine_all == 0):
                break
    
    def predict(self, x):
        y_pred = np.ones(len(x))
        for i in range(len(x)):
            if x[i] * self.w + self.b < 0:
                 y_pred[i] = 0
        
        return y_pred