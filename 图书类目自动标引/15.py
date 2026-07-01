# -*- coding: utf-8 -*-
"""15.py - 可视化分析
生成6张可视化图表"""

import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.svm import LinearSVC
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

IMG_DIR = '实验结果图片'
import os
os.makedirs(IMG_DIR, exist_ok=True)

with open('processed_data.pkl', 'rb') as f:
    data = pickle.load(f)
train_df, val_df, test_df = data['train'], data['val'], data['test']

with open('tuned_results.pkl', 'rb') as f:
    tuned = pickle.load(f)
test_results = tuned['test_results']
best_params = tuned['best_params']
cv_results = tuned['cv_results']

num_cols = ['title_char_len', 'abstract_char_len', 'abstract_word_count', 'keyword_count', 'has_abstract']

tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)
X_train_tfidf = tfidf.fit_transform(train_df['text'])
X_test_tfidf = tfidf.transform(test_df['text'])
scaler = MinMaxScaler()
X_train_num = csr_matrix(scaler.fit_transform(train_df[num_cols]))
X_test_num = csr_matrix(scaler.transform(test_df[num_cols]))
X_train = hstack([X_train_tfidf, X_train_num])
X_test = hstack([X_test_tfidf, X_test_num])

y_train = train_df['label'].values
y_test = test_df['label'].values
label_names = sorted(set(y_train) & set(y_test))
n_classes = len(label_names)

best_model = LinearSVC(C=best_params['C'], max_iter=5000, class_weight=best_params.get('class_weight'))
best_model.fit(X_train, y_train)
y_pred_test = best_model.predict(X_test)

print('=' * 60)
print('开始生成可视化图表')
print('=' * 60)

# 图1: 整体混淆矩阵
print('  图5-1: 整体混淆矩阵...')
cm = confusion_matrix(y_test, y_pred_test, labels=label_names)
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
ax.set_title('图5-1：最优模型整体混淆矩阵（测试集）', fontsize=14)
ax.set_xlabel('预测类别', fontsize=12)
ax.set_ylabel('真实类别', fontsize=12)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
step = max(1, n_classes // 15)
ticks = range(0, n_classes, step)
ax.set_xticks(ticks); ax.set_yticks(ticks)
ax.set_xticklabels([label_names[i] for i in ticks], rotation=45, fontsize=8)
ax.set_yticklabels([label_names[i] for i in ticks], fontsize=8)
plt.tight_layout()
plt.savefig('%s/图5-1_整体混淆矩阵.png' % IMG_DIR, dpi=150, bbox_inches='tight')
plt.close()

# 图2: TOP10易混淆类别
print('  图5-2: TOP10易混淆类别...')
np.fill_diagonal(cm, 0)
pairs = []
for i in range(len(label_names)):
    for j in range(len(label_names)):
        if cm[i, j] > 0:
            pairs.append((label_names[i], label_names[j], cm[i, j]))
pairs.sort(key=lambda x: x[2], reverse=True)
top10 = pairs[:10]
fig, ax = plt.subplots(figsize=(10, 6))
labels = [f"{a} -> {b}" for a, b, _ in top10]
counts = [c for _, _, c in top10]
ax.barh(range(len(labels)), counts, color='#4472C4')
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels[::-1], fontsize=10)
ax.set_xlabel('误分类样本数', fontsize=12)
ax.set_title('图5-2：TOP10易混淆类别（真实->预测）', fontsize=14)
ax.invert_yaxis()
for bar, c in zip(ax.patches, counts[::-1]):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, str(c), va='center', fontsize=10)
plt.tight_layout()
plt.savefig('%s/图5-2_TOP10易混淆类别.png' % IMG_DIR, dpi=150, bbox_inches='tight')
plt.close()

# 图3: ROC曲线
print('  图5-3: ROC曲线...')
top5_labels = [lbl for lbl, _ in train_df['label'].value_counts().head(5).items()]
try:
    y_scores = best_model.decision_function(X_test)
except:
    y_scores = best_model.predict_proba(X_test)
label_to_idx = {l: i for i, l in enumerate(label_names)}
fig, ax = plt.subplots(figsize=(8, 6))
colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000', '#5B9BD5']
for idx, lbl in enumerate(top5_labels):
    if lbl in label_to_idx:
        i = label_to_idx[lbl]
        y_bin = (y_test == lbl).astype(int)
        score = y_scores[:, i] if y_scores.ndim > 1 else y_scores
        fpr, tpr, _ = roc_curve(y_bin, score)
        ax.plot(fpr, tpr, color=colors[idx], lw=2, label='%s (AUC=%.3f)' % (lbl, auc(fpr, tpr)))
ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_xlabel('假正例率 (FPR)', fontsize=12)
ax.set_ylabel('真正例率 (TPR)', fontsize=12)
ax.set_title('图5-3：TOP5高频类别ROC曲线', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('%s/图5-3_ROC曲线.png' % IMG_DIR, dpi=150, bbox_inches='tight')
plt.close()

# 图4: TOP20特征重要性
print('  图5-4: TOP20特征重要性...')
feature_names = list(tfidf.get_feature_names_out()) + num_cols
coef = best_model.coef_
mean_abs = np.mean(np.abs(coef), axis=0)
top20_idx = np.argsort(mean_abs)[-20:][::-1]
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(range(20), mean_abs[top20_idx][::-1], color='#5B9BD5')
ax.set_yticks(range(20))
ax.set_yticklabels([feature_names[i] for i in top20_idx[::-1]], fontsize=10)
ax.set_xlabel('平均绝对系数', fontsize=12)
ax.set_title('图5-4：TOP20重要特征（平均绝对系数）', fontsize=14)
plt.tight_layout()
plt.savefig('%s/图5-4_TOP20特征重要性.png' % IMG_DIR, dpi=150, bbox_inches='tight')
plt.close()

# 图5: 模型对比柱状图
print('  图5-5: 模型对比柱状图...')
mn = list(test_results.keys())
x = np.arange(len(mn)); width = 0.2
fig, ax = plt.subplots(figsize=(12, 6))
for i, (mk, ml, mc) in enumerate(zip(['acc', 'precision', 'recall', 'f1'],
                                     ['准确率', '精确率', '召回率', 'F1值'],
                                     ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000'])):
    vals = [test_results[n][mk] for n in mn]
    bars = ax.bar(x + (i - 1.5) * width, vals, width, label=ml, color=mc)
ax.set_xlabel('模型', fontsize=12)
ax.set_ylabel('指标值', fontsize=12)
ax.set_title('图5-5：模型测试集性能对比', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(mn, fontsize=9, rotation=15)
ax.legend(fontsize=10)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig('%s/图5-5_模型对比柱状图.png' % IMG_DIR, dpi=150, bbox_inches='tight')
plt.close()

# 图6: 参数调优趋势
print('  图5-6: 参数调优趋势...')
pvals = [p['C'] for p in cv_results['params'] if p['class_weight'] == best_params.get('class_weight')]
mscores = [cv_results['mean_test_score'][i] for i, p in enumerate(cv_results['params']) if p['class_weight'] == best_params.get('class_weight')]
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(len(pvals)), mscores, 'o-', color='#4472C4', lw=2, markersize=8)
ax.fill_between(range(len(pvals)),
                [m - 0.005 for m in mscores],
                [m + 0.005 for m in mscores],
                alpha=0.15, color='#4472C4')
ax.set_xticks(range(len(pvals)))
ax.set_xticklabels([str(v) for v in pvals])
ax.set_xlabel('正则化系数 C', fontsize=12)
ax.set_ylabel('交叉验证准确率', fontsize=12)
ax.set_title('图5-6：LinearSVC参数调优趋势（class_weight=%s）' % str(best_params.get('class_weight')), fontsize=14)
ax.grid(True, alpha=0.3)
for i, (v, s) in enumerate(zip(pvals, mscores)):
    ax.annotate('%.4f' % s, (i, s), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('%s/图5-6_参数调优趋势.png' % IMG_DIR, dpi=150, bbox_inches='tight')
plt.close()

print('\n所有图表已保存至 %s/' % IMG_DIR)
