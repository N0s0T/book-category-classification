# -*- coding: utf-8 -*-
"""14.py - 模型对比与网格搜索调优
GridSearchCV搜索最优C和class_weight"""

import pandas as pd
import numpy as np
import pickle
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

with open('processed_data.pkl', 'rb') as f:
    data = pickle.load(f)
train_df, val_df, test_df = data['train'], data['val'], data['test']

num_cols = ['title_char_len', 'abstract_char_len', 'abstract_word_count', 'keyword_count', 'has_abstract']

tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)
X_train_tfidf = tfidf.fit_transform(train_df['text'])
X_val_tfidf = tfidf.transform(val_df['text'])
X_test_tfidf = tfidf.transform(test_df['text'])

scaler = MinMaxScaler()
X_train_num = csr_matrix(scaler.fit_transform(train_df[num_cols]))
X_val_num = csr_matrix(scaler.transform(val_df[num_cols]))
X_test_num = csr_matrix(scaler.transform(test_df[num_cols]))

X_train = hstack([X_train_tfidf, X_train_num])
X_val = hstack([X_val_tfidf, X_val_num])
X_test = hstack([X_test_tfidf, X_test_num])

y_train = train_df['label'].values
y_val = val_df['label'].values
y_test = test_df['label'].values

# 第1部分：多模型测试集对比
print('=' * 60)
print('第1部分: 多模型测试集对比')
print('=' * 60)

models = {
    'MultinomialNB': MultinomialNB(alpha=0.5),
    'LogisticRegression': LogisticRegression(C=1.0, max_iter=3000, solver='lbfgs', n_jobs=-1),
    'LinearSVC': LinearSVC(C=1.0, max_iter=5000)
}

test_results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_results[name] = {
        'acc': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='macro', zero_division=0)
    }
    print('  %s: Acc=%.4f, P=%.4f, R=%.4f, F1=%.4f' % (name, test_results[name]['acc'], test_results[name]['precision'], test_results[name]['recall'], test_results[name]['f1']))

# 第2部分：GridSearchCV调优LinearSVC
print('\n' + '=' * 60)
print('第2部分: LinearSVC 网格搜索调优')
print('=' * 60)

param_grid = {
    'C': [0.1, 0.5, 1.0, 2.0, 5.0],
    'class_weight': [None, 'balanced']
}

grid = GridSearchCV(
    LinearSVC(max_iter=5000),
    param_grid,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)
grid.fit(X_train, y_train)

print('最优参数: %s' % grid.best_params_)
print('交叉验证最佳准确率: %.4f' % grid.best_score_)

# 各参数组合详细结果
cv_results = grid.cv_results_
print('\n各参数组合详细结果:')
for i, params in enumerate(cv_results['params']):
    print('  C=%s, class_weight=%s: mean_acc=%.4f' % (params['C'], str(params['class_weight']), cv_results['mean_test_score'][i]))

# 最优模型测试集评估
best_model = grid.best_estimator_
y_pred_best = best_model.predict(X_test)
best_test = {
    'acc': accuracy_score(y_test, y_pred_best),
    'precision': precision_score(y_test, y_pred_best, average='macro', zero_division=0),
    'recall': recall_score(y_test, y_pred_best, average='macro', zero_division=0),
    'f1': f1_score(y_test, y_pred_best, average='macro', zero_division=0)
}
print('\n最优模型测试集: Acc=%.4f, P=%.4f, R=%.4f, F1=%.4f' % (best_test['acc'], best_test['precision'], best_test['recall'], best_test['f1']))

test_results['LinearSVC(调优)'] = best_test

with open('tuned_results.pkl', 'wb') as f:
    pickle.dump({
        'test_results': test_results,
        'best_params': grid.best_params_,
        'best_score': grid.best_score_,
        'cv_results': cv_results,
        'best_model': best_model,
        'y_pred_best': y_pred_best
    }, f)

print('\n结果已保存至 tuned_results.pkl')
