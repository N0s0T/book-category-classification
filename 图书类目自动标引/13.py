# -*- coding: utf-8 -*-
"""13.py - 模型训练
TF-IDF(8k, 1-2gram) + 5个数值特征，训练三种基线模型"""

import pandas as pd
import numpy as np
import pickle
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

with open('processed_data.pkl', 'rb') as f:
    data = pickle.load(f)
train_df, val_df = data['train'], data['val']

num_cols = ['title_char_len', 'abstract_char_len', 'abstract_word_count', 'keyword_count', 'has_abstract']

# TF-IDF向量化
tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)
X_train_tfidf = tfidf.fit_transform(train_df['text'])
X_val_tfidf = tfidf.transform(val_df['text'])

# 数值特征归一化
scaler = MinMaxScaler()
X_train_num = csr_matrix(scaler.fit_transform(train_df[num_cols]))
X_val_num = csr_matrix(scaler.transform(val_df[num_cols]))

# 合并特征
X_train = hstack([X_train_tfidf, X_train_num])
X_val = hstack([X_val_tfidf, X_val_num])

y_train = train_df['label'].values
y_val = val_df['label'].values

print('特征维度: TF-IDF=%d, 数值=%d, 总计=%d' % (X_train_tfidf.shape[1], len(num_cols), X_train.shape[1]))
print('类别数: %d, 训练集: %d, 验证集: %d' % (len(set(y_train)), len(y_train), len(y_val)))

# 训练三种基线模型
models = {
    'MultinomialNB': MultinomialNB(alpha=0.5),
    'LogisticRegression': LogisticRegression(C=1.0, max_iter=3000, solver='lbfgs', n_jobs=-1),
    'LinearSVC': LinearSVC(C=1.0, max_iter=5000)
}

print('\n基线模型验证集结果:')
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    results = {
        'acc': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_val, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_val, y_pred, average='macro', zero_division=0)
    }
    print('  %s: Acc=%.4f, P=%.4f, R=%.4f, F1=%.4f' % (name, results['acc'], results['precision'], results['recall'], results['f1']))

with open('baseline_results.pkl', 'wb') as f:
    pickle.dump({'models': models, 'tfidf': tfidf, 'scaler': scaler, 'results': results}, f)
print('\n结果已保存至 baseline_results.pkl')
