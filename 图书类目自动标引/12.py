# -*- coding: utf-8 -*-
"""12.py - 数据预处理
利用完整12列特征，增强文本构建，类别过滤>=100"""

import pandas as pd
import numpy as np
import warnings
import pickle
warnings.filterwarnings('ignore')

DATA_DIR = '第3章 数据集/data'
COLS = ['path','label','title','keywords','abstract','kw_simple','pos_tag','abstract_seg','pos_seq','topic_kw1','topic_kw2','topic_dist']

def load_data(filepath):
    df = pd.read_csv(filepath, header=None).dropna(how='all').iloc[:, :12]
    df.columns = COLS
    return df

train = load_data(f'{DATA_DIR}/train.csv')
val = load_data(f'{DATA_DIR}/val.csv')
test = load_data(f'{DATA_DIR}/test.csv')

def preprocess(df, name):
    df = df.copy()
    # 增强文本：标题 + 关键词 + 摘要 + 精简关键词
    df['text'] = (df['title'].fillna('') + ' ' +
                  df['keywords'].fillna('') + ' ' +
                  df['abstract'].fillna('') + ' ' +
                  df['kw_simple'].fillna('')).str.lower().str.strip()
    # 5个手工数值特征
    df['title_char_len'] = df['title'].fillna('').str.len()
    df['abstract_char_len'] = df['abstract'].fillna('').str.len()
    df['abstract_word_count'] = df['abstract'].fillna('').apply(lambda x: len(str(x).split()))
    df['keyword_count'] = df['keywords'].fillna('').apply(lambda x: len([k for k in str(x).split() if k.strip()]))
    df['has_abstract'] = (df['abstract'].fillna('').str.strip() != '').astype(int)
    print('[%s] 预处理完成: %d条, %d个类别' % (name, len(df), df['label'].nunique()))
    return df

train_df = preprocess(train, 'train')
val_df = preprocess(val, 'val')
test_df = preprocess(test, 'test')

# 只保留三集共有类别，且训练样本>=100
common = set(train_df['label']) & set(val_df['label']) & set(test_df['label'])
train_counts = train_df['label'].value_counts()
freq_labels = set(train_counts[train_counts >= 100].index)
common_labels = common & freq_labels
print('三集共有: %d类, 频率过滤(>=100): %d类, 最终保留: %d类' % (len(common), len(freq_labels), len(common_labels)))

train_df = train_df[train_df['label'].isin(common_labels)].reset_index(drop=True)
val_df = val_df[val_df['label'].isin(common_labels)].reset_index(drop=True)
test_df = test_df[test_df['label'].isin(common_labels)].reset_index(drop=True)

print('最终: 训练%d / 验证%d / 测试%d, %d个类别' % (len(train_df), len(val_df), len(test_df), len(common_labels)))

with open('processed_data.pkl', 'wb') as f:
    pickle.dump({'train': train_df, 'val': val_df, 'test': test_df}, f)
print('数据已保存至 processed_data.pkl')
