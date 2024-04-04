import pandas as pd
import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

##### HEALTHY VS CANCER

Train_data = 'data/Train_Set.csv'
Test_data = 'data/Test_Set.csv'

Train_dataset = pd.read_csv(Train_data)
Test_dataset = pd.read_csv(Test_data)

int_label = {'healthy': 0, 'screening stage cancer': 1,
              'early stage cancer': 1, 'mid stage cancer': 1,
              'late stage cancer': 1}

Train_dataset['Int_label'] = Train_dataset['class_label'].map(int_label)
Test_dataset['Int_label'] = Test_dataset['class_label'].map(int_label)

Train_X = Train_dataset.iloc[:, :350]
Test_X = Test_dataset.iloc[:, :350]
Train_y = Train_dataset['Int_label']
Test_y = Test_dataset['Int_label']

scaler = StandardScaler()
Train_X = scaler.fit_transform(Train_X)
Test_X = scaler.transform(Test_X)

random_forest = RandomForestClassifier(random_state=0, class_weight='balanced')
random_forest.fit(Train_X, Train_y)

Pred_y_RF = random_forest.predict(Test_X)

classification_report_RF = classification_report(Test_y, Pred_y_RF)
confusion_matrix_RF = confusion_matrix(Test_y, Pred_y_RF)
accuracy_RF = accuracy_score(Test_y, Pred_y_RF)

print('##########')
print('Using Random Forest:')
print('Healthy vs Cancer')
print(classification_report_RF)
print(confusion_matrix_RF)

y_pred_prob_RF = random_forest.predict_proba(Test_X)[:, 1]

fpr_RF, tpr_RF, thresholds_RF = roc_curve(Test_y, y_pred_prob_RF)
roc_auc_RF = auc(fpr_RF, tpr_RF)

plt.figure(figsize=(8, 6))
plt.plot(fpr_RF, tpr_RF, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_RF)
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Random Forest')
plt.legend(loc='lower right')
plt.show()

##### HEALTHY VS SCREENING STAGE CANCER

Train_data = 'data/Train_Set.csv'
Test_data = 'data/Test_Set.csv'

Train_dataset = pd.read_csv(Train_data)
Test_dataset = pd.read_csv(Test_data)

int_label = {'healthy': 0, 'screening stage cancer': 1}

Train_dataset['Int_label'] = Train_dataset['class_label'].map(int_label)
Test_dataset['Int_label'] = Test_dataset['class_label'].map(int_label)

Train_dataset = Train_dataset.dropna(subset=['Int_label']).reset_index(drop=True)
Test_dataset = Test_dataset.dropna(subset=['Int_label']).reset_index(drop=True)

Train_dataset['Int_label'] = Train_dataset['Int_label'].astype(int)
Test_dataset['Int_label'] = Test_dataset['Int_label'].astype(int)

Train_X = Train_dataset.iloc[:, :350]
Test_X = Test_dataset.iloc[:, :350]
Train_y = Train_dataset['Int_label']
Test_y = Test_dataset['Int_label']

scaler = StandardScaler()
Train_X = scaler.fit_transform(Train_X)
Test_X = scaler.transform(Test_X)

random_forest = RandomForestClassifier(random_state=0, class_weight='balanced')
random_forest.fit(Train_X, Train_y)

Pred_y_RF = random_forest.predict(Test_X)
Pred_y_RF_train = random_forest.predict(Train_X)

classification_report_RF = classification_report(Test_y, Pred_y_RF)
confusion_matrix_RF = confusion_matrix(Test_y, Pred_y_RF)
accuracy_RF = accuracy_score(Test_y, Pred_y_RF)
accuracy_RF_train = accuracy_score(Train_y, Pred_y_RF_train)

print('##########')
print('Using Random Forest:')
print('Healthy vs Screening Stage Cancer')
print(classification_report_RF)
print(confusion_matrix_RF)
print(f'Training accuracy: {accuracy_RF_train}')
print(f'Testing accuracy: {accuracy_RF}')

##### Healthy vs Early Stage Cancer

Train_data = 'data/Train_Set.csv'
Test_data = 'data/Test_Set.csv'

Train_dataset = pd.read_csv(Train_data)
Test_dataset = pd.read_csv(Test_data)

int_label = {'healthy': 0, 'early stage cancer': 1}

Train_dataset['Int_label'] = Train_dataset['class_label'].map(int_label)
Test_dataset['Int_label'] = Test_dataset['class_label'].map(int_label)

Train_dataset = Train_dataset.dropna(subset=['Int_label']).reset_index(drop=True)
Test_dataset = Test_dataset.dropna(subset=['Int_label']).reset_index(drop=True)

Train_dataset['Int_label'] = Train_dataset['Int_label'].astype(int)
Test_dataset['Int_label'] = Test_dataset['Int_label'].astype(int)

Train_X = Train_dataset.iloc[:, :350]
Test_X = Test_dataset.iloc[:, :350]
Train_y = Train_dataset['Int_label']
Test_y = Test_dataset['Int_label']

scaler = StandardScaler()
Train_X = scaler.fit_transform(Train_X)
Test_X = scaler.transform(Test_X)

random_forest = RandomForestClassifier(random_state=0, class_weight='balanced')
random_forest.fit(Train_X, Train_y)

Pred_y_RF = random_forest.predict(Test_X)
Pred_y_RF_train = random_forest.predict(Train_X)

classification_report_RF = classification_report(Test_y, Pred_y_RF)
confusion_matrix_RF = confusion_matrix(Test_y, Pred_y_RF)
accuracy_RF = accuracy_score(Test_y, Pred_y_RF)
accuracy_RF_train = accuracy_score(Train_y, Pred_y_RF_train)

print('##########')
print('Using Random Forest:')
print('Healthy vs Early Stage Cancer')
print(classification_report_RF)
print(confusion_matrix_RF)
print(f'Training accuracy: {accuracy_RF_train}')
print(f'Testing accuracy: {accuracy_RF}')