import pandas as pd
import csv
import re
import numpy as np
import math
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import matplotlib.pyplot as plt
np.random.seed(0)

class P2DatasetReader:
    def __init__(self, file_name: str) -> None:           
        self.df = pd.read_csv(file_name, sep=';', engine='python')
        self.df= self.df.dropna()

    def get_data_frame(self) -> pd.DataFrame:
        return self.df


    def remove_duplicates(self, subset: list) -> None:
        self.df = self.df.drop_duplicates(subset=[self.df.columns[i] for i in subset], keep='last')


    def keep_only_n_classes(self, n: int) -> None:
        value_counts = self.df[self.df.columns[1]].value_counts()
        elegible_indexes = value_counts.index[0:n]
        new_df = pd.DataFrame()
        for index in elegible_indexes:
            item = self.df[self.df[self.df.columns[1]] == index]
            new_df = pd.concat([new_df, item])
        
        self.df = new_df

    def keep_only_classes_with_more_than_n_instances(self, n: int) -> None:
        value_counts = self.df[self.df.columns[1]].value_counts()
        new_df = pd.DataFrame()
        for i in range(len(value_counts)):
            if value_counts[i] >= n:
                item = self.df[self.df[self.df.columns[1]] == value_counts.index[i]]
                new_df = pd.concat([new_df, item])
        
        self.df = new_df


    def value_counts(self) -> None:
        return self.df[self.df.columns[1]].value_counts()


    def get_minority_class_size(self) -> int:
        return math.ceil(min([self.df[self.df[self.df.columns[1]] == index].shape[0] for index in self.value_counts().index]))


    def downsample(self) -> None:
        min_class_size = self.get_minority_class_size()

        new_df = pd.DataFrame()
        for index in self.value_counts().index:
            item = self.df[self.df[self.df.columns[1]] == index]
            item_shape = item.shape[0]
            chosen_idx = np.random.choice(item_shape, replace=False, size=min_class_size) 

            new_df = pd.concat([new_df, item.iloc[chosen_idx]], axis=0)
            
        self.df = new_df


    def split_n_folds(self, n_folds) -> None:
        skf = StratifiedKFold(n_splits=n_folds)
        dataset = self.get_data_frame()

        X = dataset.iloc[:, 2:]
        y = dataset.iloc[:, 1]
        y = y.astype(str)
        
        for train_index, test_index in skf.split(X, y):
            yield X.iloc[train_index], pd.get_dummies(y.iloc[train_index]), X.iloc[test_index], pd.get_dummies(y.iloc[test_index])


    def class_names(self) -> None:
        y = self.get_data_frame().iloc[:, 1]
        y = y.astype(str)
        names = np.unique(y)
        return names

    def count_graph(self) -> None:
        plt.figure()
        plt.xticks(rotation=90)
        sns.countplot(x = self.df[self.df.columns[1]])
        plt.tight_layout()
        plt.show()