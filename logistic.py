import numpy as np
from scipy import optimize
from sklearn.model_selection import KFold

class Logistic():
    def __init__(self, x_train, y_train):
        print("Training data in Logistic Regression")
        self.Xtrain = np.concatenate((np.ones((x_train.shape[0], 1)), x_train), axis = 1)
        self.ytrain = y_train[:, np.newaxis]
        self.theta = np.zeros((self.Xtrain.shape[1], 1))
        print("Performing K Fold Cross Validation")
        self.lamb = self.k_fold_cross_validation(self.Xtrain, self.ytrain, 10)
        print("K Fold cross validation complete")
        self.theta = self.fit(self.Xtrain, self.ytrain, self.theta)
        
    def sigmoid(self, x):
        # Activation function used to map any real value between 0 and 1
        return 1 / (1 + np.exp(-x))

    def net_input(self, theta, x):
        # Computes the weighted sum of inputs
        return np.dot(x, theta)
    
    def probability(self, theta, x):
        # Returns the probability after passing through sigmoid
        return self.sigmoid(self.net_input(theta, x))
    
    def cost_function(self, theta, x, y):
        m = x.shape[0]
        total_cost = -(1 / m) * np.sum(y * np.log(self.probability(theta, x)) + 
                       (1 - y) * np.log(1 - self.probability(theta, x))) + self.lamb * np.linalg.norm(theta, ord = 2)
        return total_cost
    
    def gradient(self, theta, x, y):
        # Computes the gradient of the cost function at the point theta
        m = x.shape[0]
        return (1 / m) * np.dot(x.T, self.sigmoid(self.net_input(theta, x)) - y) + self.lamb * theta
    
    def flatten(t):
        t = t.reshape(1, -1)
        t = t.squeeze()
        return t

    def fit(self, x, y, theta):
        opt_weights = optimize.fmin_tnc(func=self.cost_function, x0=theta, 
                                        fprime=self.gradient, args=(x, y.flatten()))
        return opt_weights[0]
    
    def predict(self, x):
        x = np.concatenate((np.ones((x.shape[0], 1)), x), axis = 1)
        theta = self.theta
        return (self.probability(theta, x) >= 0.5).astype(int).flatten()
    
    def k_fold_cross_validation(self, x, y, K):
        kf = KFold(n_splits=K) # Define the split - into K folds 
        kf.get_n_splits(x) # returns the number of splitting iterations in the cross-validator
        min_lambda_error = float("inf")
        min_lambda = 0
        
    
        for lam in np.arange(0.1, 1, 0.1):
            self.lamb = lam
            error = 0
            epsilon = 0
            for train_index, test_index in kf.split(x):
                Dx_trn, Dx_tst = x[train_index], x[test_index]
                Dy_trn, Dy_tst = y[train_index], y[test_index]
                
                #Calculate theeta for current fold
                theeta = self.fit(Dx_trn, Dy_trn, np.zeros((Dx_trn.shape[1], 1)))
                
                #Calculate error for this lambda value at current fold
                error += self.cost_function(theeta, Dx_tst, Dy_tst)
                
            epsilon = error/K
            
            #Get lambda and theeta for minimum error
            if(epsilon < min_lambda_error):
                min_lambda_error = epsilon
                min_lambda = lam
        return min_lambda