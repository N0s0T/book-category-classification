# 导入所需库
import pandas as pd
import numpy as np
import jieba
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import os

# 设置绘图中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
tqdm.pandas()

# ===================== 路径配置（你的项目结构） =====================
base_path = "第3章 数据集"

# ===================== 任务1：导入数据 =====================
train = pd.read_csv(f"{base_path}/data/train.csv", encoding="utf-8")
print("[OK] 数据集真实列名：", train.columns.tolist())

# 固定列名（根据你真实的列名直接写死）
col_file = train.columns[0]  # 文件路径
col_category = train.columns[1]  # 类目标签
col_title = train.columns[2]  # 标题（书名）
col_keyword = train.columns[3]  # 关键词
col_abstract = train.columns[4]  # 摘要


# ===================== 任务2：数据清洗（去重+格式统一） =====================
def data_clean(df):
    print(f"\n去重前：{len(df)}")
    df = df.drop_duplicates(subset=[col_title, col_abstract], keep="first")
    df[col_title] = df[col_title].astype(str).str.strip()
    df[col_abstract] = df[col_abstract].astype(str).str.strip()
    df[col_keyword] = df[col_keyword].astype(str).str.strip()
    print(f"去重后：{len(df)}")
    return df


# ===================== 任务3：缺失值处理 =====================
def deal_missing(df):
    print("\n缺失值统计：")
    print(df.isnull().sum())
    df[col_title] = df[col_title].fillna("")
    df[col_abstract] = df[col_abstract].fillna("")
    df[col_keyword] = df[col_keyword].fillna("")
    print("[OK] 缺失值已全部填充为空")
    return df


# ===================== 任务4：异常值处理 =====================
def deal_outlier(df):
    df = df[df[col_title].str.len() >= 2]
    df = df[df[col_abstract].str.len() >= 5]
    print(f"[OK] 异常值过滤完成，剩余：{len(df)}")
    return df


# ===================== 任务5：特征工程 =====================
stop_words_path = f"{base_path}/stop.txt"
stopwords = set([line.strip() for line in open(stop_words_path, encoding="utf-8")])


def cut_words(text):
    words = jieba.lcut(str(text).strip())
    return " ".join([w for w in words if w not in stopwords and len(w) > 1 and not w.isdigit()])


def feature_engineer(df):
    print("\n开始分词...")
    df["title_cut"] = df[col_title].progress_apply(cut_words)
    df["abstract_cut"] = df[col_abstract].progress_apply(cut_words)
    df["all_text"] = df["title_cut"] + " " + df["abstract_cut"] + " " + df[col_keyword]

    le = LabelEncoder()
    df["label"] = le.fit_transform(df[col_category])

    tfidf = TfidfVectorizer(max_features=3000)
    X = tfidf.fit_transform(df["all_text"]).toarray()
    print(f"[OK] TF-IDF特征维度：{X.shape}")
    return df, X


# ===================== 任务6：EDA可视化图表 =====================
def eda(df):
    print("\n正在生成图表...")

    # 1. 类目分布
    plt.figure(figsize=(12, 5))
    df[col_category].value_counts().plot(kind="bar", color="#4285F4")
    plt.title("图书类目分布")
    plt.tight_layout()
    plt.show()

    # 2. 标题长度分布
    df["title_len"] = df[col_title].str.len()
    plt.figure(figsize=(10, 4))
    sns.histplot(df["title_len"], bins=30, kde=True, color="orange")
    plt.title("标题长度分布")
    plt.show()

    # 3. 关键词数量分布
    df["kw_count"] = df[col_keyword].apply(lambda x: len(str(x).split()))
    plt.figure(figsize=(10, 4))
    sns.countplot(x=df["kw_count"], color="green")
    plt.title("关键词数量分布")
    plt.show()

    # 4. 热力图
    num_df = df[["title_len", "kw_count"]]
    plt.figure(figsize=(6, 4))
    sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm")
    plt.title("特征相关性热力图")
    plt.show()

    # 5. 高频词
    all_words = " ".join(df["all_text"]).split()
    top20 = Counter(all_words).most_common(20)
    top_df = pd.DataFrame(top20, columns=["词", "频次"])
    plt.figure(figsize=(12, 5))
    sns.barplot(x="频次", y="词", data=top_df, palette="viridis")
    plt.title("高频词TOP20")
    plt.tight_layout()
    plt.show()


# ===================== 主程序入口 =====================
if __name__ == "__main__":
    print("[Start] 开始执行6大任务...")
    train = data_clean(train)
    train = deal_missing(train)
    train = deal_outlier(train)
    train, tfidf_feat = feature_engineer(train)
    eda(train)

    # 保存结果
    out_path = os.path.join(base_path, "train_final_result.csv")
    train.to_csv(out_path, index=False, encoding="utf-8")

    print("\n 全部6个任务完成！")
    print(f" 结果已保存到：{out_path}")