from experiment.multiclass.albert import ALBERT_exp
from experiment.multiclass.bert import BERT_exp
from experiment.multiclass.lstm import LSTM_exp
from experiment.multiclass.roberta import ROBERTA_exp
from experiment.multiclass.electra import ELECTRA_exp
from experiment.multiclass.p2 import P2_exp


# p2_exp = P2_exp(name='DLP-P2', 
#                     dataset='fasttext-crawl-300d-2M.csv', 
#                     epochs=10,
#                     folds=10).run()
# p2_exp = P2_exp(name='DL-P2', 
#                     dataset='dl-fasttext-crawl-300d-2M.csv', 
#                     epochs=10,
#                     folds=10).run()


lstm_exp = LSTM_exp(name='DLP-LSTM-64', 
                    dataset='dlp-multiclass-dataset.csv', 
                    train_index=3,
                    output_dim=64, 
                    epochs=1,
                    folds=10).run()
lstm_exp = LSTM_exp(name='DL-LSTM-64', 
                    dataset='dl-multiclass-dataset.csv', 
                    train_index=3,
                    output_dim=64, 
                    epochs=10,
                    folds=10).run()


lstm_exp = LSTM_exp(name='DLP-LSTM-128', 
                    dataset='dlp-multiclass-dataset.csv', 
                    train_index=3,
                    output_dim=128, 
                    epochs=10,
                    folds=10).run()
lstm_exp = LSTM_exp(name='DL-LSTM-128', 
                    dataset='dl-multiclass-dataset.csv', 
                    train_index=3,
                    output_dim=128, 
                    epochs=10,
                    folds=10).run()


lstm_exp = LSTM_exp(name='DLP-LSTM-256', 
                    dataset='dlp-multiclass-dataset.csv', 
                    train_index=3,
                    output_dim=256, 
                    epochs=10,
                    folds=10).run()
lstm_exp = LSTM_exp(name='DL-LSTM-256', 
                    dataset='dl-multiclass-dataset.csv', 
                    train_index=3,
                    output_dim=256, 
                    epochs=10,
                    folds=10).run()





bert_exp = BERT_exp(name='DLP-BERT-Tiny', 
                    model_name='small_bert/bert_en_uncased_L-2_H-128_A-2', 
                    dataset='dlp-multiclass-dataset.csv', 
                    train_index=3,
                    learning_rate=0.0003,
                    batch_size=8,
                    epochs=100,
                    folds=10,
                    trainable=False).run()
bert_exp = BERT_exp(name='DL-BERT-Tiny', 
                    model_name='small_bert/bert_en_uncased_L-2_H-128_A-2', 
                    dataset='dl-multiclass-dataset.csv', 
                    train_index=3, 
                    learning_rate=0.0003,
                    batch_size=8,
                    epochs=100,
                    folds=10,
                    trainable=False).run()


bert_exp = BERT_exp(name='DLP-BERT-Mini', 
                    model_name='small_bert/bert_en_uncased_L-4_H-256_A-4', 
                    dataset='dlp-multiclass-dataset.csv', 
                    train_index=3,
                    learning_rate=0.0001,
                    batch_size=16, 
                    epochs=100,
                    folds=10,
                    trainable=False).run()
bert_exp = BERT_exp(name='DL-BERT-Mini', 
                    model_name='small_bert/bert_en_uncased_L-4_H-256_A-4', 
                    dataset='dl-multiclass-dataset.csv', 
                    learning_rate=0.0001,
                    batch_size=16,
                    train_index=3, 
                    epochs=100,
                    folds=10,
                    trainable=False).run()


bert_exp = BERT_exp(name='DLP-BERT-Small', 
                    model_name='small_bert/bert_en_uncased_L-4_H-512_A-8', 
                    dataset='dlp-multiclass-dataset.csv', 
                    train_index=3,
                    learning_rate=0.00005,
                    batch_size=32, 
                    epochs=100,
                    folds=10,
                    trainable=False).run()
bert_exp = BERT_exp(name='DL-BERT-Small', 
                    model_name='small_bert/bert_en_uncased_L-4_H-512_A-8', 
                    dataset='dl-multiclass-dataset.csv', 
                    learning_rate=0.00005,
                    batch_size=32,
                    train_index=3, 
                    epochs=100,
                    folds=10,
                    trainable=False).run()


bert_exp = BERT_exp(name='DLP-BERT-Medium', 
                    model_name='small_bert/bert_en_uncased_L-8_H-512_A-8', 
                    dataset='dlp-multiclass-dataset.csv', 
                    train_index=3,
                    learning_rate=0.00003,
                    batch_size=64, 
                    epochs=100,
                    folds=10,
                    trainable=False).run()
bert_exp = BERT_exp(name='DL-BERT-Medium', 
                    model_name='small_bert/bert_en_uncased_L-8_H-512_A-8', 
                    dataset='dl-multiclass-dataset.csv', 
                    learning_rate=0.00003,
                    batch_size=64,
                    train_index=3, 
                    epochs=100,
                    folds=10,
                    trainable=False).run()


bert_exp = BERT_exp(name='DLP-BERT-Base', 
                    model_name='bert_en_uncased_L-12_H-768_A-12', 
                    dataset='dlp-multiclass-dataset.csv', 
                    learning_rate=0.00002,
                    batch_size=32,
                    train_index=3, 
                    epochs=100,
                    folds=10,
                    trainable=False).run()
bert_exp = BERT_exp(name='DL-BERT-Base', 
                    model_name='bert_en_uncased_L-12_H-768_A-12', 
                    dataset='dl-multiclass-dataset.csv', 
                    learning_rate=0.00002,
                    batch_size=32,
                    train_index=3, 
                    epochs=100,
                    folds=10,
                    trainable=False).run()

electra_exp = ELECTRA_exp(name='DLP-ELECTRA-Small', 
                    model_name='electra_en_small', 
                    dataset='dlp-multiclass-dataset.csv', 
                    train_index=3,
                    learning_rate=0.00005,
                    batch_size=32, 
                    epochs=10,
                    folds=10,
                    trainable=False).run()
electra_exp = ELECTRA_exp(name='DL-ELECTRA-Small', 
                    model_name='electra_en_small', 
                    dataset='dl-multiclass-dataset.csv', 
                    learning_rate=0.00005,
                    batch_size=32,
                    train_index=3, 
                    epochs=10,
                    folds=10,
                    trainable=False).run()

albert_exp = ALBERT_exp(name='DLP-ALBERT-Base', 
                    model_name='albert_en_base', 
                    dataset='dlp-multiclass-dataset.csv', 
                    train_index=3,
                    learning_rate=0.00002,
                    batch_size=32, 
                    epochs=10,
                    folds=10,
                    trainable=False).run()
albert_exp = ALBERT_exp(name='DL-ALBERT-Base', 
                    model_name='albert_en_base', 
                    dataset='dl-multiclass-dataset.csv', 
                    learning_rate=0.00002,
                    batch_size=32,
                    train_index=3, 
                    epochs=10,
                    folds=10,
                    trainable=False).run()


roberta_exp = ROBERTA_exp(name='DLP-ROBERTA-Base', 
                    model_name='roberta_en_base', 
                    dataset='dlp-multiclass-dataset.csv', 
                    train_index=3,
                    learning_rate=0.00002,
                    batch_size=32, 
                    epochs=10,
                    folds=10,
                    trainable=False).run()
roberta_exp = ROBERTA_exp(name='DL-ROBERTA-Base', 
                    model_name='roberta_en_base', 
                    dataset='dl-multiclass-dataset.csv', 
                    learning_rate=0.00002,
                    batch_size=32,
                    train_index=3, 
                    epochs=10,
                    folds=10,
                    trainable=False).run()


electra_exp = ELECTRA_exp(name='DLP-ELECTRA-Base', 
                    model_name='electra_en_base', 
                    dataset='dlp-multiclass-dataset.csv', 
                    train_index=3,
                    learning_rate=0.00002,
                    batch_size=32, 
                    epochs=10,
                    folds=10).run()
electra_exp = ELECTRA_exp(name='DL-ELECTRA-Base', 
                    model_name='electra_en_base', 
                    dataset='dl-multiclass-dataset.csv', 
                    learning_rate=0.00002,
                    batch_size=32,
                    train_index=3, 
                    epochs=10,
                    folds=10).run()



