import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

df_AB=pd.read_csv('df_AB--RotationEncodingBL62.txt_EncodingMatrix.txt', sep="\t", header=None)
print(len(df_AB))

df_AB = df_AB[df_AB.iloc[:, 6] == 'MusMusculus']

value_counts = df_AB.iloc[:, 10].value_counts()
values_to_keep = value_counts[value_counts >= 2].index
print(values_to_keep)
df_AB = df_AB[df_AB.iloc[:, 10].isin(values_to_keep)]
print(len(df_AB))

print(np.shape(df_AB))
data_AB_human = df_AB.iloc[:,14:]
labels, unique_values = pd.factorize(df_AB.iloc[:, 10])
print(len(labels))
mapping_dict = {i: val for i, val in enumerate(unique_values)}

data_ab_human = data_AB_human.to_numpy()
print(np.shape(data_ab_human))

n = len(labels)
train_idx, test_idx, y_train, y_test = train_test_split(
    np.arange(n), labels, test_size=0.2, stratify=labels)

# df_ab_bh_full = df_ab_bh_full[df_ab_bh_full['epitope'].isin(values_to_keep)]
# print(len(df_ab_bh_full))

X_train = data_ab_human[train_idx]
X_test = data_ab_human[test_idx]


k_range = list(range(1, 50))

# 创建 KNN 模型
knn = KNeighborsClassifier()

# 创建参数网格
param_grid = dict(n_neighbors=k_range)

# 设置网格搜索
grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')

# 进行网格搜索
grid.fit(X_train, y_train)

# knn = KNeighborsClassifier(n_neighbors=3)

# # 训练模型
# knn.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
print("Best cross-validation score: {:.2f}".format(grid.best_score_))

# 使用最佳参数的模型在测试集上进行预测
best_knn = grid.best_estimator_
y_pred = best_knn.predict(X_test)
y_pred1 = best_knn.predict(X_train)
accuracy1 = accuracy_score(y_train, y_pred1)
print(f"Train set accuracy: {accuracy1:.2f}")

# 计算并打印准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Test set accuracy: {accuracy:.2f}")


