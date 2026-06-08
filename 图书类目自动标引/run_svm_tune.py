import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import warnings
import sys
warnings.filterwarnings('ignore')

# 保存输出到文件
output_lines = []

def log(msg):
    print(msg)
    output_lines.append(msg)

# 加载数据
log("加载数据...")
train = pd.read_csv("第3章 数据集/train_final_result.csv")
val = pd.read_csv("第3章 数据集/data/val.csv")
val = val.fillna("")

log(f"训练集样本数: {len(train)}")
log(f"验证集样本数: {len(val)}")

# 列名
col_cat = val.columns[1]
col_title = val.columns[2]
col_key = val.columns[3]
col_abs = val.columns[4]

# 构造文本特征
X_train = train["all_text"].astype(str)
X_val = val[col_title].astype(str) + " " + val[col_abs].astype(str) + " " + val[col_key].astype(str)

# 使用第2列作为训练标签（和14.py一致）
y_train = train.iloc[:, 1]
y_val = val[col_cat]

# 标签编码
le = LabelEncoder()
le.fit(y_train)

# 过滤验证集中训练集未出现的类别
mask = y_val.isin(le.classes_)
y_val_filtered = y_val[mask]
X_val_filtered = X_val[mask]
y_val_encoded = le.transform(y_val_filtered)
y_train_encoded = le.transform(y_train)

log(f"有效验证样本数: {len(y_val_filtered)}")

# TF-IDF特征提取
tfidf = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

log("构建TF-IDF特征...")
X_train_vec = tfidf.fit_transform(X_train)
X_val_vec = tfidf.transform(X_val_filtered)

# SVM调优
log("\n[调优1] SVM - LinearSVC")
log("-" * 60)
param_grid_svm = {"C": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]}
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid_svm = GridSearchCV(
    LinearSVC(max_iter=3000),
    param_grid_svm,
    cv=skf,
    scoring='accuracy',
    n_jobs=-1
)
grid_svm.fit(X_train_vec, y_train_encoded)

log("C参数搜索:")
for params, mean_score, std_score in zip(
    grid_svm.cv_results_['params'],
    grid_svm.cv_results_['mean_test_score'],
    grid_svm.cv_results_['std_test_score']
):
    marker = " <-- 最优" if params == grid_svm.best_params_ else ""
    log(f"  C={params['C']:<8} 准确率={mean_score:.4f} (+/- {std_score:.4f}){marker}")

best_svm = grid_svm.best_estimator_
pred_best_svm = best_svm.predict(X_val_vec)
acc_best_svm = accuracy_score(y_val_encoded, pred_best_svm)
log(f"\n最优参数: {grid_svm.best_params_}")
log(f"交叉验证最优准确率: {grid_svm.best_score_:.4f}")
log(f"验证集准确率: {acc_best_svm:.2%}")

# 写入文件
with open("svm_tune_output.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))