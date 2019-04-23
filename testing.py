import numpy as np
from logistic import Logistic
from merge_data import get_data_skin, create_breast_data
from confusion_matrix import confusionMatrix
from svm import SVM
from ROC import plot_roc_curve
from nn import neural_net
from cnn import cnn

def logistic_breast():
    print("\nLogistic Regression for Breast Cancer data:\n")
    x_train, x_test, y_train, y_test = create_breast_data()
    logistic = Logistic(x_train, y_train)
    y_pred = logistic.predict(x_train)
    print("\nTraining Classification accuracy: ")
    print(100 - 100*np.sum(np.abs(y_pred - y_train))/y_pred.shape[0])
    confusionMatrix(y_train, y_pred)
    y_pred = logistic.predict(x_test)
    print("\nTesting Classification accuracy: ")
    print(100 - 100*np.sum(np.abs(y_pred - y_test))/y_pred.shape[0])
    confusionMatrix(y_test, y_pred)
    print("ROC Curve: ")
    plot_roc_curve(y_test, y_pred)
    
def logistic_skin():
    print("\nLogistic Regression for Skin Cancer data:\n")
    x_train, x_test, y_train, y_test = get_data_skin()
    logistic = Logistic(x_train, y_train)
    y_pred = logistic.predict(x_train)
    print("\nTraining Classification accuracy: ")
    print(100 - 100*np.sum(np.abs(y_pred - y_train))/y_pred.shape[0])
    confusionMatrix(y_train, y_pred)
    y_pred = logistic.predict(x_test)
    print("\nTesting Classification accuracy: ")
    print(100 - 100*np.sum(np.abs(y_pred - y_test))/y_pred.shape[0])
    confusionMatrix(y_test, y_pred)
    print("ROC Curve: ")
    plot_roc_curve(y_test, y_pred)
    
def nn_breast():
    print("\nFeed-forward Neural Nets for Breast Cancer:\n")
    neural_net()
    
def nn_skin():
    print("\nFeed-forward Neural Nets for Skin Cancer:\n")
    neural_net('skin')
    
def cnn_skin():
    print("\nConvolutional Neural Nets for Skin Cancer:\n")
    cnn()

def svm_breast():
    print("\nSVM Classification for Breast Cancer data:\n")
    x_train, x_test, y_train, y_test = create_breast_data()
    svm = SVM(x_train, np.where(y_train==0, -1, y_train), 100, 0.01)
    y_pred = svm.predict(x_train)
    print("\nTraining Classification accuracy: ")
    print(100 - 100*np.sum(np.abs(y_pred - y_train))/y_pred.shape[0])
    confusionMatrix(y_train, y_pred)
    y_pred = svm.predict(x_test)
    print("\nTesting Classification accuracy: ")
    print(100 - 100*np.sum(np.abs(y_pred - y_test))/y_pred.shape[0])
    confusionMatrix(y_test, y_pred)
    print("ROC Curve: ")
    plot_roc_curve(y_test, y_pred)
    
def svm_skin():
    print("\nSVM Classification for Skin Cancer data:\n")
    x_train, x_test, y_train, y_test = get_data_skin()
    svm = SVM(x_train, np.where(y_train==0, -1, y_train), 100, 0.01)
    y_pred = svm.predict(x_train)
    print("\nTraining Classification accuracy: ")
    print(100 - 100*np.sum(np.abs(y_pred - y_train))/y_pred.shape[0])
    confusionMatrix(y_train, y_pred)
    y_pred = svm.predict(x_test)
    print("\nTesting Classification accuracy: ")
    print(100 - 100*np.sum(np.abs(y_pred - y_test))/y_pred.shape[0])
    confusionMatrix(y_test, y_pred)
    print("ROC Curve: ")
    plot_roc_curve(y_test, y_pred)
    
def main():
    logistic_breast()
    logistic_skin()
    nn_breast()
    nn_skin()
    cnn_skin()
    svm_breast()
    svm_skin()
    
main()