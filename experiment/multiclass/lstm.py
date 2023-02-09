import numpy as np
from sklearn import datasets
from model.multiclass.LSTM import LSTM_model
from evaluation.confusion_matrix import ConfusionMatrix
from evaluation.accuracy_loss import Accuracy_Loss
from evaluation.roc import ROC
from evaluation.classification_report import ClassificationReport
from dataset_reader.multiclass_dataset_reader import MultiClassDatasetReader
# from official.nlp import optimization  # to create AdamW optimizer
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


import random
random.seed(0)

class LSTM_exp():
    epochs = 0
    folds = 0
    dataset = ''
    name = ''
    train_index = 0
    output_dim = 0

    def __init__(self, train_index: int, dataset: str, epochs: int, folds: int, name: str, output_dim: int) -> None:
        self.epochs = epochs
        self.folds = folds
        self.dataset = dataset
        self.name = name
        self.train_index = train_index
        self.output_dim = output_dim

    def get_encoder(self, comments):
        encoder_layer = layers.experimental.preprocessing.TextVectorization(output_mode='int')
        encoder_layer.adapt(np.asarray(comments))
        return encoder_layer

    def run(self):    
        dataset = MultiClassDatasetReader(self.dataset)
        dataset.remove_duplicates([self.train_index])
        dataset.keep_only_classes_with_more_than_n_instances(self.folds)
        # dataset.keep_only_n_classes(30)
        # dataset.downsample()
        # dataset.count_graph()
        print(dataset.value_counts())
        fold = 0

        all_y = []
        all_p = []
        all_train_acc = []
        all_train_loss = []
        all_val_acc = []
        all_val_loss = []

        enconder_layer = self.get_encoder(dataset.get_data_frame().iloc[:, self.train_index])
        print(enconder_layer.vocabulary_size())
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

        roc = ROC(classes=dataset.class_names(), title=self.name)

        for x_train, y_train, x_test, y_test in dataset.split_n_folds(self.folds):
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1)

            train_comment = x_train.iloc[:, self.train_index-1]
            test_comment = x_test.iloc[:, self.train_index-1]
            eval_comment = x_val.iloc[:, self.train_index-1]

            print(train_comment)
            print(test_comment)
            print(eval_comment)

            model = LSTM_model(enconder_layer, enconder_layer.vocabulary_size(), self.output_dim, len(dataset.class_names())).get_model()

            model.compile(optimizer = 'Adam',
                                loss='categorical_crossentropy',
                                metrics=['accuracy'])
            print(model.summary())
            hist = model.fit(train_comment, y_train, epochs=self.epochs, validation_data = (eval_comment, y_val), callbacks=[callback])
            # hist = model.fit(train_comment, y_train, epochs=self.epochs, validation_data = (eval_comment, y_val))

            predictions = model.predict(test_comment)
            roc.append(np.asarray(y_test), np.asarray(predictions))

            y = []
            p = []

            y = [np.argmax(t, axis=0) for t in np.asarray(y_test)]
            p = [np.argmax(t, axis=0) for t in np.asarray(predictions)]
            [all_y.append(v) for v in y]
            [all_p.append(v) for v in p]

            confusion_matrix = ConfusionMatrix(y, p, f'{self.name}_FOLD_{fold+1}', dataset.class_names())
            confusion_matrix.save_fig()

            classification_report = ClassificationReport(y, p, dataset.class_names(), f'{self.name}_FOLD_{fold+1}')
            classification_report.save()

            history_dict = hist.history
            train_acc  = history_dict['accuracy']
            train_loss = history_dict['loss']
            val_acc    = history_dict['val_accuracy']
            val_loss   = history_dict['val_loss']

            all_train_acc.append(train_acc)
            all_train_loss.append(train_loss)
            all_val_acc.append(val_acc)
            all_val_loss.append(val_loss)

            curve = Accuracy_Loss(train_acc, train_loss, val_acc, val_loss, f'{self.name}_AC_FOLD_{fold+1}')
            curve.save_fig()

            fold+=1

        roc.save()
        confusion_matrix = ConfusionMatrix(all_y, all_p, self.name, dataset.class_names())
        confusion_matrix.save_fig()

        classification_report = ClassificationReport(all_y, all_p, dataset.class_names(), f'{self.name}')
        classification_report.save()

        all_train_acc = np.sum(all_train_acc, axis=0) / len(all_train_acc)
        all_train_loss = np.sum(all_train_loss, axis=0) / len(all_train_loss)
        all_val_acc = np.sum(all_val_acc, axis=0) / len(all_val_acc)
        all_val_loss = np.sum(all_val_loss, axis=0) / len(all_val_loss)

        curve = Accuracy_Loss(all_train_acc, all_train_loss, all_val_acc, all_val_loss, f'{self.name}_AC')
        curve.save_fig()
        