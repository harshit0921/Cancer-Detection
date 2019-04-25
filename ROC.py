import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve  
from sklearn.metrics import roc_auc_score 

def plot_roc_curve(target, predicted):
    auc = roc_auc_score(target, predicted)
    print('AUC: %.2f' % auc)
    fpr, tpr, thresholds = roc_curve(target, predicted)
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
