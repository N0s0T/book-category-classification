import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, label_binarize
from itertools import cycle

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

print("=" * 80)
print("图书分类模型可视化分析 (改进版)")
print("=" * 80)

# 1. 加载数据（和14.py保持一致）
print("\n[步骤1] 数据加载与预处理")
train = pd.read_csv("第3章 数据集/train_final_result.csv")
val = pd.read_csv("第3章 数据集/data/val.csv")
val = val.fillna("")

col_cat = val.columns[1]
col_title = val.columns[2]
col_key = val.columns[3]
col_abs = val.columns[4]

X_train = train["all_text"].astype(str)
X_val = val[col_title].astype(str) + " " + val[col_abs].astype(str) + " " + val[col_key].astype(str)
y_train = train["label"]
y_val = val[col_cat]

# 标签编码 + 过滤未知类别
train_categories = train.iloc[:, 1]
le = LabelEncoder()
le.fit(train_categories)
mask = y_val.isin(le.classes_)
y_val_filtered = y_val[mask]
X_val_filtered = X_val[mask]
y_val_encoded = le.transform(y_val_filtered)

print(f"训练集样本数: {len(train)}")
print(f"验证集样本数: {len(y_val_filtered)} (过滤了{len(y_val) - len(y_val_filtered)}个)")

# 2. 改进版TF-IDF特征提取
print("\n[步骤2] TF-IDF特征提取")
tfidf = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)
X_train_vec = tfidf.fit_transform(X_train)
X_val_vec = tfidf.transform(X_val_filtered)
print(f"特征维度: {X_train_vec.shape[1]}")

# 3. 训练模型（在训练集上训练，在验证集上评估）
print("\n[步骤3] 模型训练")
svm = LinearSVC(C=0.1, max_iter=2000)
svm.fit(X_train_vec, y_train)
pred_svm = svm.predict(X_val_vec)  # 在验证集上预测！
print(f"SVM验证集准确率: {accuracy_score(y_val_encoded, pred_svm):.2%}")

# ===================== 1. 绘制混淆矩阵（改进版） =====================
print("\n[图表1] 绘制混淆矩阵")
cm = confusion_matrix(y_val_encoded, pred_svm)

plt.figure(figsize=(12, 10))
im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('SVM模型混淆矩阵 (验证集)', fontsize=14, pad=20)
plt.colorbar(im, label='样本数')

# 由于类别太多(200+)，不显示具体类别标签，只显示刻度
plt.xticks([])
plt.yticks([])

plt.xlabel('预测类别', fontsize=12)
plt.ylabel('真实类别', fontsize=12)

# 添加对角线标注（正确预测区域）
plt.plot([0, cm.shape[0]], [0, cm.shape[1]], 'r--', linewidth=2, alpha=0.5, label='正确预测')
plt.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig("混淆矩阵_改进版.png", dpi=300, bbox_inches='tight')
print("  → 已保存: 混淆矩阵_改进版.png")
plt.show()

# ===================== 2. 绘制混淆最严重的类别 =====================
print("\n[图表2] 绘制混淆最严重的类别")
cm_copy = cm.copy()
np.fill_diagonal(cm_copy, 0)  # 对角线上是正确预测，不计入混淆
confusion_total = cm_copy.sum(axis=1)  # 每个类别的混淆总数
top_confused_idx = np.argsort(confusion_total)[-10:][::-1]  # 混淆最严重的10个类别

plt.figure(figsize=(14, 8))
cm_top = cm[np.ix_(top_confused_idx, top_confused_idx)]
im = plt.imshow(cm_top, interpolation='nearest', cmap=plt.cm.Reds)
plt.title('混淆最严重的TOP10类别 (验证集)', fontsize=14, pad=20)
plt.colorbar(im, label='样本数')
plt.xticks(range(len(top_confused_idx)), 
           [str(le.classes_[i])[:12] for i in top_confused_idx], 
           rotation=45, ha='right', fontsize=10)
plt.yticks(range(len(top_confused_idx)), 
           [str(le.classes_[i])[:12] for i in top_confused_idx], 
           fontsize=10)
plt.xlabel('预测类别', fontsize=12)
plt.ylabel('真实类别', fontsize=12)

# 在格子里标注数字
for i in range(len(top_confused_idx)):
    for j in range(len(top_confused_idx)):
        plt.text(j, i, str(cm_top[i, j]), 
                ha="center", va="center", 
                color="white" if cm_top[i, j] > cm_top.max()/2 else "black")

plt.tight_layout()
plt.savefig("混淆矩阵_TOP10.png", dpi=300, bbox_inches='tight')
print("  → 已保存: 混淆矩阵_TOP10.png")
plt.show()

# ===================== 3. 绘制多分类ROC曲线（改进版） =====================
print("\n[图表3] 绘制ROC曲线")
n_classes = len(le.classes_)
y_bin = label_binarize(y_val_encoded, classes=range(n_classes))
y_score = svm.decision_function(X_val_vec)  # 在验证集上计算！

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    if np.sum(y_bin[:, i]) > 0:  # 只处理有样本的类别
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

# 选择样本数最多的5个类别来展示
class_counts = np.sum(y_bin, axis=0)
top_classes = np.argsort(class_counts)[-5:][::-1]

plt.figure(figsize=(10, 8))
colors = cycle(['aqua', 'darkorange', 'limegreen', 'blue', 'red'])
for i, color in zip(top_classes, colors):
    if i in roc_auc:
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{le.classes_[i][:15]} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正例率 (FPR)', fontsize=12)
plt.ylabel('真正例率 (TPR)', fontsize=12)
plt.title('多分类ROC曲线 (验证集 - 样本数最多的5个类别)', fontsize=14)
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("ROC曲线_改进版.png", dpi=300, bbox_inches='tight')
print("  → 已保存: ROC曲线_改进版.png")
plt.show()

# ===================== 4. 绘制特征重要性（改进版） =====================
print("\n[图表4] 绘制特征重要性")
feature_names = tfidf.get_feature_names_out()

# SVM是one-vs-rest，coef_是(n_classes, n_features)
# 应该平均所有类别的权重绝对值
coef_mean = np.mean(np.abs(svm.coef_), axis=0)  # 平均所有类别
top_idx = np.argsort(coef_mean)[-20:][::-1]  # 取权重绝对值前20特征
top_coef = coef_mean[top_idx]
top_features = [feature_names[i] for i in top_idx]

plt.figure(figsize=(14, 8))
bars = plt.barh(range(len(top_features)), top_coef, color='#4A90E2')
plt.yticks(range(len(top_features)), top_features, fontsize=11)
plt.xlabel('平均特征权重 (绝对值)', fontsize=12)
plt.ylabel('特征词', fontsize=12)
plt.title('TOP20 重要特征 (SVM所有类别平均)', fontsize=14, pad=20)
plt.gca().invert_yaxis()  # 把重要的放上面
plt.grid(axis='x', alpha=0.3)

# 在柱子上标注数值
for i, (bar, coef) in enumerate(zip(bars, top_coef)):
    width = bar.get_width()
    plt.text(width + 0.005, bar.get_y() + bar.get_height()/2,
             f'{coef:.4f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig("特征重要性_改进版.png", dpi=300, bbox_inches='tight')
print("  → 已保存: 特征重要性_改进版.png")
plt.show()

# ===================== 5. 绘制模型性能对比图 =====================
print("\n[图表5] 绘制模型性能对比")
# 先训练几个模型对比
models = [
    ("朴素贝叶斯", MultinomialNB(alpha=0.5)),
    ("SVM", LinearSVC(C=0.1, max_iter=2000)),
    ("逻辑回归", __import__('sklearn.linear_model').linear_model.LogisticRegression(C=0.5, max_iter=1000))
]

acc_list = []
for name, model in models:
    model.fit(X_train_vec, y_train)
    pred = model.predict(X_val_vec)
    acc_list.append(accuracy_score(y_val_encoded, pred))

plt.figure(figsize=(10, 6))
bars = plt.bar([m[0] for m in models], acc_list, color=['#7ED321', '#4A90E2', '#F5A623'])
plt.ylim(0, max(acc_list) * 1.2)
plt.ylabel('准确率', fontsize=12)
plt.title('模型性能对比 (验证集)', fontsize=14, pad=20)
plt.grid(axis='y', alpha=0.3)

# 标注数值
for bar, acc in zip(bars, acc_list):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.005,
             f'{acc:.2%}', ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig("模型对比_改进版.png", dpi=300, bbox_inches='tight')
print("  → 已保存: 模型对比_改进版.png")
plt.show()

# ===================== 6. SVM超参数调优 =====================
print("\n[步骤4] SVM超参数调优")
param_grid = {"C": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]}
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid = GridSearchCV(LinearSVC(max_iter=3000), param_grid, cv=skf, scoring='accuracy', n_jobs=-1)
grid.fit(X_train_vec, y_train)

print("最优参数:", grid.best_params_)
print("交叉验证最优准确率:", f"{grid.best_score_:.2%}")
best_svm = grid.best_estimator_
pred_best = best_svm.predict(X_val_vec)
print("验证集准确率:", f"{accuracy_score(y_val_encoded, pred_best):.2%}")

# 绘制调优过程
plt.figure(figsize=(10, 6))
cs = [p['C'] for p in grid.cv_results_['params']]
scores = grid.cv_results_['mean_test_score']
stds = grid.cv_results_['std_test_score']

plt.errorbar(cs, scores, yerr=stds, fmt='o-', color='#4A90E2', capsize=5, linewidth=2, markersize=8)
plt.axvline(grid.best_params_['C'], color='#F5A623', linestyle='--', linewidth=2, label=f'最优C={grid.best_params_["C"]}')
plt.xscale('log')
plt.xlabel('正则化参数 C (对数刻度)', fontsize=12)
plt.ylabel('交叉验证准确率', fontsize=12)
plt.title('SVM超参数调优过程', fontsize=14, pad=20)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("SVM超参数调优_改进版.png", dpi=300, bbox_inches='tight')
print("  → 已保存: SVM超参数调优_改进版.png")
plt.show()

print("\n" + "=" * 80)
print("所有图表绘制完成！共生成6个图表：")
print("  1. 混淆矩阵_改进版.png")
print("  2. 混淆矩阵_TOP10.png")
print("  3. ROC曲线_改进版.png")
print("  4. 特征重要性_改进版.png")
print("  5. 模型对比_改进版.png")
print("  6. SVM超参数调优_改进版.png")
print("=" * 80)
