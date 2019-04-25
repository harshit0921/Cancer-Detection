from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def confusionMatrix(target, predicted):
    matrix = confusion_matrix(target, predicted)
    target_names = ['Benign', 'Malignant']
    sns.heatmap(matrix.T, square=True, annot=True, fmt='d', cbar=True,
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()
    TP = matrix[1][1]
    TN = matrix[0][0]
    FP = matrix[1][0]
    FN = matrix[0][1]
    return TP, TN, FP, FN
