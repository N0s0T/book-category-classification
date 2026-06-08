import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

print("=" * 60)
print("改进版图书分类模型 v2.0")
print("=" * 60)

# 1. 加载数据
print("\n[Step 1] Loading data...")
train = pd.read_csv("第3章 数据集/train_final_result.csv")
val = pd.read_csv("第3章 数据集/data/val.csv")
val = val.fillna("")

print(f"  Training samples: {len(train)}")
print(f"  Validation samples: {len(val)}")

# 2. 列名
col_cat_val = val.columns[1]
col_title = val.columns[2]
col_key = val.columns[3]
col_abs = val.columns[4]

# 3. 构造文本
X_val = val[col_title].astype(str) + " " + val[col_abs].astype(str) + " " + val[col_key].astype(str)
y_val = val[col_cat_val]

X_train = train["all_text"].astype(str)
y_train = train["label"]
train_categories = train[train.columns[1]]

# 4. 标签编码
le = LabelEncoder()
le.fit(train_categories)

# 5. 过滤验证集中未见过的类别
mask = y_val.isin(le.classes_)
y_val_filtered = y_val[mask]
X_val_filtered = X_val[mask]
y_val_encoded = le.transform(y_val_filtered)

print(f"\n  Valid validation samples: {len(y_val_filtered)}")
print(f"  Excluded (unseen categories): {len(y_val) - len(y_val_filtered)}")

# 6. 改进版 TF-IDF
print("\n[Step 2] Building improved TF-IDF features...")

# 改进点 1: 优化TF-IDF参数
tfidf = TfidfVectorizer(
    max_features=8000,      # 从3000增加到8000
    ngram_range=(1, 2),    # 使用unigram + bigram
    min_df=2,              # 去除稀有词
    max_df=0.95,           # 去除过于常见的词
    sublinear_tf=True       # 使用对数TF，更稳定
)

X_train_vec = tfidf.fit_transform(X_train)
X_val_vec = tfidf.transform(X_val_filtered)

print(f"  Feature dimension: {X_train_vec.shape[1]}")
print(f"  Using bigrams: (1,2)")
print(f"  Using sublinear TF: True")

# 7. 训练模型
print("\n[Step 3] Training models...")

# 改进点 2: 调整朴素贝叶斯参数
print("\n  3.1 Training Naive Bayes (improved)...")
nb = MultinomialNB(alpha=0.5)  # alpha从1.0调整到0.5
nb.fit(X_train_vec, y_train)
pred_nb = nb.predict(X_val_vec)
acc_nb = accuracy_score(y_val_encoded, pred_nb)
print(f"      Accuracy: {acc_nb:.2%}")

# 改进点 3: SVM正则化
print("\n  3.2 Training SVM (regularized)...")
svc = LinearSVC(
    C=0.1,              # 从C=1.0改为C=0.1，减少过拟合
    max_iter=2000,      # 增加迭代次数
    dual=True
)
svc.fit(X_train_vec, y_train)
pred_svc = svc.predict(X_val_vec)
acc_svc = accuracy_score(y_val_encoded, pred_svc)
print(f"      Accuracy: {acc_svc:.2%}")

# 8. 结果对比
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

baseline_acc = 0.2405  # 原始准确率

print(f"\n{'':30s} {'Accuracy':<12} {'Improvement':<12} {'Status'}")
print("-" * 70)

results = [
    ("Baseline (Old Model)", baseline_acc, 0.0, "Reference"),
    ("Naive Bayes (Improved)", acc_nb, acc_nb - baseline_acc, "Improved" if acc_nb > baseline_acc else "Same"),
    ("SVM (Regularized)", acc_svc, acc_svc - baseline_acc, "Best!" if acc_svc == max(acc_nb, acc_svc) else "")
]

for name, acc, imp, status in results:
    print(f"{name:30s} {acc:>10.2%}  {imp:>+10.2%}  {status}")

best_acc = max(acc_nb, acc_svc)
best_name = "SVM (Regularized)" if acc_svc >= acc_nb else "Naive Bayes"

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)
print(f"\n  Model: {best_name}")
print(f"  Accuracy: {best_acc:.2%}")
print(f"  Improvement: +{(best_acc - baseline_acc)*100:.1f}%")
print(f"  Relative improvement: {(best_acc/baseline_acc - 1)*100:.1f}%")

# 9. 改进总结
print("\n" + "=" * 60)
print("IMPROVEMENTS APPLIED")
print("=" * 60)

print("""
  1. TF-IDF Optimization:
     - Features: 3000 -> 8000
     - N-gram: (1,1) -> (1,2) [Added bigrams]
     - Added min_df=2 [Remove rare words]
     - Added max_df=0.95 [Remove common words]
     - Added sublinear_tf=True [More stable]

  2. Naive Bayes:
     - Alpha: 1.0 -> 0.5 [Better smoothing]

  3. SVM Regularization:
     - C parameter: 1.0 -> 0.1 [Reduce overfitting]
     - Max iterations: 1000 -> 2000 [Better convergence]
""")

print("=" * 60)
print("Model evaluation completed successfully!")
print("=" * 60)
