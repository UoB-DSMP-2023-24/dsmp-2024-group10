from tcrdist.repertoire import TCRrep
from tcrdist.plotting import plot_pairings, _write_svg
import pandas as pd
from tcrdist.repertoire import TCRrep
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.utils import resample



df = pd.read_csv('vdjdb.txt',sep='\t')
df = df.dropna()
df=df[df['vdjdb.score']!=0]   #
print("len(df): ", len(df))
df1 = df.drop(['reference.id', 'method', 'meta','cdr3fix','vdjdb.score','web.method','web.method.seq','web.cdr3fix.nc','web.cdr3fix.unmp'], axis=1)

# df_aa= df1[df1['gene']=='TRA']
# df_bb= df1[df1['gene']=='TRB']

df_a_b = df1.loc[(df1['complex.id']!=0)]
df_a_b=df_a_b.rename(columns={'antigen.epitope': 'epitope', 'v.segm': 'v_a_gene','j.segm': 'j_a_gene','cdr3': 'cdr3_a_aa','species':'subject'})
df_a_b = df_a_b.drop([ 'mhc.a','mhc.b','mhc.class','antigen.gene','antigen.species'], axis=1)
print("len(df_a_b): ", len(df_a_b))
df_paired_alpha=df_a_b.loc[df_a_b.gene=='TRA']
df_paired_beta=df_a_b.loc[df_a_b.gene=='TRB']

df_paired_ab=pd.merge(df_paired_alpha, df_paired_beta, on='complex.id', how='outer')
df_paired_ab1=df_paired_ab.dropna()
df_paired_ab1 = df_paired_ab1.drop([ 'gene_x','subject_x','epitope_x','gene_y','complex.id'], axis=1)
df_paired_ab1=df_paired_ab1.rename(columns={'cdr3_a_aa_x':'cdr3_a_aa','cdr3_a_aa_y':'cdr3_b_aa','v_a_gene_x': 'v_a_gene', 'j_a_gene_x': 'j_a_gene','v_a_gene_y': 'v_b_gene','cdr3': 'cdr3_a_aa','j_a_gene_y':'j_b_gene','epitope_y':'epitope'})


df_ab_bh=df_paired_ab1 [df_paired_ab1 ['subject_y']=='HomoSapiens']
#print(df_ab_bh)
df_ab_bh=df_ab_bh.drop(['subject_y'], axis=1)
df_ab_bh = df_ab_bh.groupby(list(df_ab_bh.columns)).size().reset_index(name='count')
print(len(df_ab_bh))
# print(df_ab_bh.head())
df_ab_bh_full = df_ab_bh.loc[df_ab_bh.index.repeat(df_ab_bh['count'])]
print(len(df_ab_bh_full))

tr_bh_ab = TCRrep(cell_df = df_ab_bh,
            organism = 'human',
            chains = ['alpha','beta'],
            db_file = 'alphabeta_gammadelta_db.tsv')
# # print(tr_bh_ab.pw_beta)
# print(type(tr_bh_ab.pw_cdr3_b_aa))
# # print(tr_bh_ab.pw_alpha)
# # print(np.shape(tr_bh_ab.pw_cdr3_a_aa))
distance_matrix = tr_bh_ab.pw_cdr3_b_aa


index_repeats = df_ab_bh['count'].values
expanded_distance_matrix = np.repeat(distance_matrix, index_repeats, axis=0)
expanded_distance_matrix = np.repeat(expanded_distance_matrix, index_repeats, axis=1)
print(np.shape(expanded_distance_matrix))

value_counts = df_ab_bh_full['epitope'].value_counts()
values_to_keep = value_counts[value_counts >= 5].index
df_ab_bh_full = df_ab_bh_full[df_ab_bh_full['epitope'].isin(values_to_keep)]
print(len(df_ab_bh_full))

kept_indices = df_ab_bh_full.index.tolist()
filtered_distance_matrix = expanded_distance_matrix[np.ix_(kept_indices, kept_indices)]
print(np.shape(filtered_distance_matrix))

df_ab_bh_full['epitope_encoded'], unique_labels = pd.factorize(df_ab_bh_full['epitope'])
df_ab_bh_labels = np.array(list(df_ab_bh_full['epitope_encoded']))

n = len(df_ab_bh_labels)
train_idx, test_idx, y_train, y_test = train_test_split(
    np.arange(n), df_ab_bh_labels, test_size=0.2, stratify=df_ab_bh_labels)

X_train = filtered_distance_matrix[train_idx][:, train_idx]
X_test = filtered_distance_matrix[test_idx][:, train_idx]


k_range = list(range(1, 500))

# 创建 KNN 模型
knn = KNeighborsClassifier(metric='precomputed')

# 创建参数网格
param_grid = dict(n_neighbors=k_range)

# 设置网格搜索
grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')

# 进行网格搜索
grid.fit(X_train, y_train)

knn = KNeighborsClassifier(metric='precomputed', n_neighbors=3)

# 训练模型
knn.fit(X_train, y_train)

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


y_scores = best_knn.predict_proba(X_test)
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

# # 获取测试集的预测概率
# y_scores = best_knn.predict_proba(X_test)


# # 假设unique_labels包含所有独特的类别标签
# unique_labels = np.unique(df_ab_bh_labels)  # 确保这是正确的类别标签集
# n_classes = len(unique_labels)

# # 二值化y_test，如果已经是二值化的可以跳过这步
# y_test_binarized = label_binarize(y_test, classes=np.arange(n_classes))

# # 预测得分（确保y_scores是使用predict_proba得到的）
# y_scores = best_knn.predict_proba(X_test)  # 这应该已经在前面的代码中完成了

# # 计算每个类别的ROC曲线和AUC
# fpr = dict()
# tpr = dict()
# roc_auc = dict()

# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_scores[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])

# # 绘制所有ROC曲线
# plt.figure(figsize=(8, 6))
# colors = iter(plt.cm.rainbow(np.linspace(0, 1, n_classes)))

# for i, color in zip(range(n_classes), colors):
#     plt.plot(fpr[i], tpr[i], color=color, lw=2,
#              label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

# plt.plot([0, 1], [0, 1], 'k--', lw=2)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic for multi-class')
# plt.legend(loc="lower right")
# plt.show()

# mean_fpr = np.linspace(0, 1, 100)
# tprs = []
# aucs = []

# # 计算100次bootstrap的ROC AUC
# for i in range(100):
#     # Bootstrap抽样
#     indices = resample(np.arange(len(y_scores)), replace=True)
#     if len(np.unique(y_test[indices])) < 2:
#         # 如果抽样得到的标签不足两类，则跳过当前循环
#         continue

#     # 计算ROC曲线
#     fpr, tpr, thresholds = roc_curve(y_test[indices], y_scores[indices])
#     roc_auc = auc(fpr, tpr)
#     tprs.append(np.interp(mean_fpr, fpr, tpr))
#     tprs[-1][0] = 0.0
#     aucs.append(roc_auc)

# mean_tpr = np.mean(tprs, axis=0)
# mean_tpr[-1] = 1.0
# mean_auc = auc(mean_fpr, mean_tpr)
# std_auc = np.std(aucs)
# std_tpr = np.std(tprs, axis=0)

# # 绘制平均ROC曲线
# plt.plot(mean_fpr, mean_tpr, color='blue',
#          label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
#          lw=2, alpha=.8)

# # 绘制置信区间阴影
# tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
# tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
# plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
#                  label=r'$\pm$ 1 std. dev.')

# plt.plot([0, 1], [0, 1], 'k--', lw=2)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic with Confidence Interval')
# plt.legend(loc="lower right")
# plt.show()