# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2024/6/11 17:40
coding with comment！！！
"""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")


# 创建一个虚拟的二分类数据集
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练GBDT模型
gbdt = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbdt.fit(X_train, y_train)

# 使用GBDT模型对训练集和测试集进行转换，得到新的特征
X_train_gbdt = gbdt.apply(X_train)[:, :, 0]

print(X_train_gbdt)

X_test_gbdt = gbdt.apply(X_test)[:, :, 0]

# 初始化LR模型
lr = LogisticRegression()

# 使用GBDT的输出作为LR的输入进行训练
lr.fit(X_train_gbdt, y_train)

# 在测试集上进行预测
y_pred = lr.predict(X_test_gbdt)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

