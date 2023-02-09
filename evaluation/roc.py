import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve, roc_auc_score
import numpy as np

class ROC():
    title = ''
    classes = []

    def __init__(self, classes, title) -> None:
        self.y_true = np.empty((0, len(classes)))
        self.y_pred = np.empty((0, len(classes)))
        self.classes = classes
        self.title = title


    # def show(self):
    #     roc = roc_auc_score(self.y_true, self.y_pred)
    #     print(roc)

    def show(self):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        n_classes = len(self.classes)
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(self.y_true[:, i], self.y_pred[:,i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
        
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(self.y_true.ravel(), self.y_pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        lw = 2
        plt.figure()
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
            linestyle=":",
            linewidth=4,
        )

        plt.plot(
            fpr["macro"],
            tpr["macro"],
            label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
            linestyle=":",
            linewidth=4,
        )

        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(self.title)
        plt.legend(loc="lower right")
        plt.show()
        

    def append(self, y_true, y_pred):
        self.y_true = np.append(self.y_true, y_true, axis=0)
        self.y_pred = np.append(self.y_pred, y_pred, axis=0)

    def save(self):
        with open(f'{self.title}_y_true.txt', 'w+') as f:
            np.savetxt(f, self.y_true, fmt='%.4f')

        with open(f'{self.title}_y_pred.txt','w+') as f:
            np.savetxt(f, self.y_pred, fmt='%.4f')
          
