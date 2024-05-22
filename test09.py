import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder

# Sample datasets; make sure to define these or load them appropriately before running the script.

df_AB = pd.read_csv("GIANA/df_AB--RotationEncodingBL62.txt_EncodingMatrix.txt", sep='\t')


# # datasets = [df_AA_Human, df_AB_Human, df_BB_Human]
# # dataset_names = ['df_AA_Human', 'df_AB_Human', 'df_BB_Human']
# #datasets = [df_AA,df_AB, df_BB]
# #dataset_names = ['df_AA', "df_AB", "df_BB"]

# # Define the key you are interested in
# key = "antigen.epitope"

# for dataset, dataset_name in zip(datasets, dataset_names):
#     print("Dataset:", dataset_name)
#     X = dataset.iloc[:, 8:104]  # Ensure these indices are correct based on your dataset's structure
#     y = dataset[key]

#     # Get the top 10 highest-frequency antigens
#     top_antigens = y.value_counts().nlargest(10).index

#     # Filter dataset to include only samples with the top 10 antigens
#     filtered_dataset = dataset[dataset[key].isin(top_antigens)]
#     X_filtered = filtered_dataset.iloc[:, 8:104]
#     y_filtered = filtered_dataset[key]

#     # Convert the target variable to binary format (numeric)
#     label_encoder = LabelEncoder()
#     y_encoded = label_encoder.fit_transform(y_filtered)

#     # Split data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_encoded, test_size=0.2, random_state=42)

#     # Initialize and train Random Forest Classifier with OvR strategy
#     rf_classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=250, random_state=42))
#     rf_classifier.fit(X_train, y_train)

#     # Predict probabilities for the test set
#     y_pred_proba = rf_classifier.predict_proba(X_test)

#     # Calculate ROC curve and AUC for each class
#     fpr = dict()
#     tpr = dict()
#     roc_auc = dict()
#     for i in range(len(label_encoder.classes_)):
#         fpr[i], tpr[i], _ = roc_curve((y_test == i).astype(int), y_pred_proba[:, i])
#         roc_auc[i] = roc_auc_score((y_test == i).astype(int), y_pred_proba[:, i])

#     # Plot ROC curve for each class
#     plt.figure(figsize=(10, 8))
#     for i in range(len(label_encoder.classes_)):
#         plt.plot(fpr[i], tpr[i], label=f'ROC curve (area = {roc_auc[i]:0.2f}) for class {label_encoder.classes_[i]}')
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title(f'ROC Curve for {dataset_name}')
#     plt.legend(loc="lower right")
#     plt.show()

#     # Calculate macro-average ROC-AUC score
#     macro_roc_auc = roc_auc_score(label_binarize(y_test, classes=range(len(label_encoder.classes_))), y_pred_proba, average='macro')
#     print(f"Macro-Average ROC-AUC Score for Dataset {dataset_name}: {macro_roc_auc}")

#     # Performance metrics
#     y_pred = rf_classifier.predict(X_test)
#     precision = precision_score(y_test, y_pred, average='weighted')
#     recall = recall_score(y_test, y_pred, average='weighted')
#     accuracy = accuracy_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred, average='weighted')

#     # Display performance metrics
#     print(f"Precision for Dataset {dataset_name}: {precision}")
#     print(f"Recall for Dataset {dataset_name}: {recall}")
#     print(f"Accuracy for Dataset {dataset_name}: {accuracy}")
#     print(f"F1 Score for Dataset {dataset_name}: {f1}")