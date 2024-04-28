import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

df_AB=pd.read_csv('df_AB--RotationEncodingBL62.txt_EncodingMatrix.txt', sep="\t", header=None)
print(len(df_AB))

df_AB = df_AB[df_AB.iloc[:, 6] == 'MusMusculus']

value_counts = df_AB.iloc[:, 10].value_counts()
values_to_keep = value_counts[value_counts >= 5].index
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

n_classes = len(set(y_test))

# 二值化y_test，如果已经是二值化的可以跳过这步
y_test_binarized = label_binarize(y_test, classes=np.arange(n_classes))
y_scores = best_knn.predict_proba(X_test)
# 计算微平均ROC曲线和AUC
fpr, tpr, _ = roc_curve(y_test_binarized.ravel(), y_scores.ravel())
roc_auc = auc(fpr, tpr)

# 绘制微平均ROC曲线
plt.figure(figsize=(8, 6))
lw = 2
plt.plot(fpr, tpr, color='blue', linestyle='-', linewidth=lw,
         label='Micro-average ROC curve (area = {0:0.2f})'.format(roc_auc))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Micro-average ROC Curve across all classes')
plt.legend(loc="lower right")
plt.show()


precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

report = classification_report(y_test, y_pred, target_names=[mapping_dict[i] for i in range(n_classes)])
print(report)