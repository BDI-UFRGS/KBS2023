import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn


class ConfusionMatrix():

    df_cm = None
    labels = None
    name = None
    class_names = []
    def __init__(self, y, p, name, class_names) -> None:
        self.name = name
        self.class_names = class_names
        conf_mat = confusion_matrix(y, p)
        group_counts = ['{0:0.0f}'.format(value) for value in
                    conf_mat.flatten()]

        conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
        
        group_percentages = ['{0:.2%}'.format(value) for value in
                        conf_mat.flatten()]


        labels = [f'{v1}\n{v2}' for v1, v2 in
            zip(group_counts, group_percentages)]

        labels = np.asarray(labels).reshape(len(class_names), len(class_names))

        df_cm = pd.DataFrame(conf_mat, index = [i for i in class_names], columns = [i for i in class_names])
        
        self.df_cm = df_cm
        self.labels = labels

    def save_fig(self) -> None:
        plt.figure(figsize=(len(self.class_names) + 3, len(self.class_names) + 1))
        plt.title(f'{self.name}')

        sn.heatmap(self.df_cm, annot=self.labels, cmap='Blues', fmt='', vmin=0, vmax=1)
        plt.tight_layout()
        plt.savefig('%s.png' % self.name)
        plt.close()
        	

class MultiLabelConfusionMatrix():
    
    def __init__(self, y, p, name, class_names) -> None:
        fig, ax = plt.subplots(3, 9, figsize=(12, 7))
    
        for axes, cfs_matrix, label in zip(ax.flatten(), np.asarray(multilabel_confusion_matrix(y, p)), class_names):
            self.__print_confusion_matrix(cfs_matrix, axes, label, ["N", "Y"])
        
        fig.tight_layout()
        plt.show()
        plt.close()

    # def save_fig(self) -> None:
    #     plt.figure(figsize=(len(self.class_names) + 3, len(self.class_names) + 1))
    #     plt.title(f'{self.name}')

    #     sn.heatmap(self.df_cm, annot=self.labels, cmap='Blues', fmt='', vmin=0, vmax=1)
    #     plt.tight_layout()
    #     plt.savefig('%s.png' % self.name)


    def __print_confusion_matrix(self, confusion_matrix, axes, class_label, class_names):
        
        df_cm = pd.DataFrame(
            confusion_matrix, index=class_names, columns=class_names,
        )

        try:
            heatmap = sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='', cbar=False, ax=axes)
        except ValueError:
            raise ValueError("Confusion matrix values must be integers.")
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
        axes.set_ylabel('True label')
        axes.set_xlabel('Predicted label')
        axes.set_title(class_label)