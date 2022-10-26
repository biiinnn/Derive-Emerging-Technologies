# -*- coding: utf-8 -*-
"""
Created on Wed May 25 16:33:03 2022

@author: yebin
"""
#%% library import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%% data
patent = pd.read_csv("../data/vacant/textvector/tfidf.csv")

# pat_num = patent['patent_number']
# title = patent['patent_title']
# abstract = patent['patent_abstract']
aafc = patent['AAFC']
sns.histplot(aafc, bins=160)
plt.title('Number of Forward Citation')
plt.xlabel(None)
plt.ylabel(None)
plt.show()

aafc.describe()
#%%
# # text 데이터만 가져오기
# text = title + '. ' + abstract
# patent['text'] = text
pr = patent.loc[patent['AAFC']>49, 'AAFC']
npr = patent.loc[patent['AAFC']<=49, 'AAFC']

sns.histplot(pr)
plt.title('Promising Patent AAFC')
plt.show()

pr.describe()

# 0,1 분류
prom = []
for i in range(len(patent)):
    if patent['AAFC'][i] <= 49:
        prom.append(0)
    else:
        prom.append(1)
        
patent['promising'] = prom



tfidf = patent.iloc[:,1:-2]
tfidf = tfidf.to_numpy()
#df_data = pd.concat([patent['text'], patent['promising']], axis=1)

# train_ = df_data
# print(train_.isnull().values.any())

#%% test
test_ = pd.read_csv("../data/vacant/textvector/vacant_text.csv")
# TF-IDF 값 계산
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
# 문서-단어 행렬 
document_term_matrix = vect.fit_transform(test_.text)       
# TF (Term Frequency)
tf = pd.DataFrame(document_term_matrix.toarray()) 
# IDF (Inverse Document Frequency)
D = len(tf)
df = tf.astype(bool).sum(axis=0)
idf = np.log((D+1) / (df+1)) + 1             
# TF-IDF (Term Frequency-Inverse Document Frequency)
tfidf_t = tf * idf                      
tfidf_t = tfidf_t / np.linalg.norm(tfidf_t, axis=1, keepdims=True)
# from sklearn.feature_extraction.text import CountVectorizer
# tdmvector = CountVectorizer()
# X_test_tdm = tdmvector.fit_transform(test_.text)
# print(X_test_tdm.shape)

# from sklearn.feature_extraction.text import TfidfTransformer
# tfidf_transformer = TfidfTransformer()
# tfidfv_t = tfidf_transformer.fit_transform(X_test_tdm)
# print(tfidfv_t.shape)

#%% BoW tfidf
# from sklearn.feature_extraction.text import CountVectorizer
# tdmvector = CountVectorizer()
# X_train_tdm = tdmvector.fit_transform(df_data.text)
# print(X_train_tdm.shape)

# from sklearn.feature_extraction.text import TfidfTransformer
# tfidf_transformer = TfidfTransformer()
# tfidfv = tfidf_transformer.fit_transform(X_train_tdm)
# print(tfidfv.shape)
#%% train vali split
from sklearn.model_selection import train_test_split
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(tfidf, patent['promising'].values, random_state=42, test_size=0.3)

#%% oversampling
from imblearn.over_sampling import SMOTE

print("Number transactions train_inputs dataset: ", train_inputs.shape)
print("Number transactions train_labels dataset: ", train_labels.shape)
print("Number transactions validation_inputs dataset: ", validation_inputs.shape)
print("Number transactions validation_labels dataset: ", validation_labels.shape)

print("Before OverSampling, counts of label '1' in train: {}".format(sum(train_labels==1)))
print("Before OverSampling, counts of label '0' in train: {} \n".format(sum(train_labels==0)))

print("Before OverSampling, counts of label '1' in test: {}".format(sum(validation_labels==1)))
print("Before OverSampling, counts of label '0' in test: {} \n".format(sum(validation_labels==0)))

# Oversampling
sm = SMOTE(random_state=42, k_neighbors=3)
train_inputs_res, train_labels_res = sm.fit_resample(train_inputs, train_labels.ravel())

print('After OverSampling, the shape of train_inputs_res: {}'.format(train_inputs_res.shape))

print('After OverSampling, the shape of train_labels_res: {} \n'.format(train_labels_res.shape))

print("After OverSampling, counts of label '1' in train_res: {}".format(sum(train_labels_res==1)))
print("After OverSampling, counts of label '0' in train_res: {}".format(sum(train_labels_res==0)))

#%% scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

scaler = MinMaxScaler()

train_inputs_res_sc = scaler.fit_transform(train_inputs_res)

validation_inputs_sc = scaler.transform(validation_inputs)

#%% calculate class weighting
from sklearn.utils.class_weight import compute_class_weight

weighting = compute_class_weight(class_weight='balanced', classes=[0,1], y=patent['promising'].values)
print(weighting)

#%% Naive bayes
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import BernoulliNB, ComplementNB
nb = ComplementNB()
nb.fit(train_inputs_res, train_labels_res)

print('Training Accuracy : %.3f'%nb.score(train_inputs_res, train_labels_res))
print('Test Accuracy : %.3f'%nb.score(validation_inputs, validation_labels))

pred_nb = nb.predict(validation_inputs)

#%% Classification report
print('Classification Report (NB):')
print(classification_report(validation_labels, pred_nb, labels=[0,1], digits=4))

#%% Confusion matrix
cm = confusion_matrix(validation_labels, pred_nb, labels=[0,1])
ax = plt.subplot()

sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")

ax.set_title('Confusion Matrix - Naive Bayes')
ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')

ax.xaxis.set_ticklabels(['Non-Promising', 'Promising'])
ax.yaxis.set_ticklabels(['Non-Promising', 'Promising'])

#%% ROC
# plot ROC Curve

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(validation_labels, pred_nb)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0,1], [0,1], 'k--' )
plt.rcParams['font.size'] = 12
plt.title('ROC curve for Naive Bayes Classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.show()

# compute ROC AUC

from sklearn.metrics import roc_auc_score

ROC_AUC = roc_auc_score(validation_labels, pred_nb)
print('ROC AUC : {:.4f}'.format(ROC_AUC))
#%% test-nb
test_nb = nb.predict(tfidf_t)

#%% Logistic Regression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression


log = LogisticRegression()
log.fit(train_inputs_res, train_labels_res)

print('Training Accuracy : %.3f'%log.score(train_inputs_res, train_labels_res))
print('Test Accuracy : %.3f'%log.score(validation_inputs, validation_labels))

pred_log = log.predict(validation_inputs)

#%% Classification report
print('Classification Report (LR):')
print(classification_report(validation_labels, pred_log, labels=[0,1], digits=4))

#%% Confusion matrix
cm = confusion_matrix(validation_labels, pred_log, labels=[0,1])
ax = plt.subplot()

sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")

ax.set_title('Confusion Matrix - Logistic Regression')
ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')

ax.xaxis.set_ticklabels(['Non-Promising', 'Promising'])
ax.yaxis.set_ticklabels(['Non-Promising', 'Promising'])

#%% ROC
# plot ROC Curve

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(validation_labels, pred_log)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0,1], [0,1], 'k--' )
plt.rcParams['font.size'] = 12
plt.title('ROC curve for Logistic Regression')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.show()

# compute ROC AUC

from sklearn.metrics import roc_auc_score

ROC_AUC = roc_auc_score(validation_labels, pred_log)
print('ROC AUC : {:.4f}'.format(ROC_AUC))
#%% test-log
test_log = log.predict(tfidf_t)

#%% SVC
from sklearn import svm

svc = svm.SVC()
svc.fit(train_inputs_res_sc, train_labels_res)
print('Training Accuracy : %.3f'%svc.score(train_inputs_res_sc, train_labels_res))
print('Test Accuracy : %.3f'%svc.score(validation_inputs_sc, validation_labels))

pred_svc = svc.predict(validation_inputs_sc)

#%% Classification report
print('Classification Report (SVC):')
print(classification_report(validation_labels, pred_svc, labels=[0,1], digits=4))

#%% Confusion matrix
cm = confusion_matrix(validation_labels, pred_svc, labels=[0,1])
ax = plt.subplot()

sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")

ax.set_title('Confusion Matrix - SVC')

ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')

ax.xaxis.set_ticklabels(['Non-Promising', 'Promising'])
ax.yaxis.set_ticklabels(['Non-Promising', 'Promising'])

#%%
fpr, tpr, thresholds = roc_curve(validation_labels, pred_svc)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0,1], [0,1], 'k--' )
plt.rcParams['font.size'] = 12
plt.title('ROC curve for SVC')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.show()

# compute ROC AUC

from sklearn.metrics import roc_auc_score

ROC_AUC = roc_auc_score(validation_labels, pred_svc)
print('ROC AUC : {:.4f}'.format(ROC_AUC))

#%% test-svc
test_svc = svc.predict(tfidf_t)





#%% Naive bayes
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import BernoulliNB, ComplementNB
nb2 = ComplementNB(class_prior=weighting)
nb2.fit(train_inputs_res, train_labels_res)

print('Training Accuracy : %.3f'%nb2.score(train_inputs_res, train_labels_res))
print('Test Accuracy : %.3f'%nb2.score(validation_inputs, validation_labels))

pred_nb2 = nb2.predict(validation_inputs)

#%% Classification report
print('Classification Report (NB):')
print(classification_report(validation_labels, pred_nb2, labels=[0,1], digits=4))

#%% Confusion matrix
cm = confusion_matrix(validation_labels, pred_nb2, labels=[0,1])
ax = plt.subplot()

sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")

ax.set_title('Confusion Matrix - Naive Bayes')
ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')

ax.xaxis.set_ticklabels(['Non-Promising', 'Promising'])
ax.yaxis.set_ticklabels(['Non-Promising', 'Promising'])

#%% ROC
# plot ROC Curve

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(validation_labels, pred_nb2)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0,1], [0,1], 'k--' )
plt.rcParams['font.size'] = 12
plt.title('ROC curve for Naive Bayes Classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.show()

# compute ROC AUC

from sklearn.metrics import roc_auc_score

ROC_AUC = roc_auc_score(validation_labels, pred_nb2)
print('ROC AUC : {:.4f}'.format(ROC_AUC))
#%% test-nb
test_nb2 = nb2.predict(tfidf_t)

#%% Logistic Regression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
weight = {0:0.50368623, 1:68.32}
log2 = LogisticRegression(class_weight=weight)
log2.fit(train_inputs_res, train_labels_res)

print('Training Accuracy : %.3f'%log2.score(train_inputs_res, train_labels_res))
print('Test Accuracy : %.3f'%log2.score(validation_inputs, validation_labels))

pred_log2 = log2.predict(validation_inputs)

#%% Classification report
print('Classification Report (LR):')
print(classification_report(validation_labels, pred_log2, labels=[0,1], digits=4))

#%% Confusion matrix
cm = confusion_matrix(validation_labels, pred_log2, labels=[0,1])
ax = plt.subplot()

sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")

ax.set_title('Confusion Matrix - Logistic Regression')
ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')

ax.xaxis.set_ticklabels(['Non-Promising', 'Promising'])
ax.yaxis.set_ticklabels(['Non-Promising', 'Promising'])

#%% ROC
# plot ROC Curve

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(validation_labels, pred_log2)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0,1], [0,1], 'k--' )
plt.rcParams['font.size'] = 12
plt.title('ROC curve for Logistic Regression')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.show()

# compute ROC AUC

from sklearn.metrics import roc_auc_score

ROC_AUC = roc_auc_score(validation_labels, pred_log2)
print('ROC AUC : {:.4f}'.format(ROC_AUC))
#%% test-log
test_log2 = log2.predict(tfidf_t)

#%% SVC
from sklearn import svm

svc2 = svm.SVC(class_weight=weight)
svc2.fit(train_inputs_res_sc, train_labels_res)
print('Training Accuracy : %.3f'%svc2.score(train_inputs_res_sc, train_labels_res))
print('Test Accuracy : %.3f'%svc2.score(validation_inputs_sc, validation_labels))

pred_svc2 = svc2.predict(validation_inputs)

#%% Classification report
print('Classification Report (SVC):')
print(classification_report(validation_labels, pred_svc2, labels=[0,1], digits=4))

#%% Confusion matrix
cm = confusion_matrix(validation_labels, pred_svc2, labels=[0,1])
ax = plt.subplot()

sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")

ax.set_title('Confusion Matrix - SVC')

ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')

ax.xaxis.set_ticklabels(['Non-Promising', 'Promising'])
ax.yaxis.set_ticklabels(['Non-Promising', 'Promising'])

#%%
fpr, tpr, thresholds = roc_curve(validation_labels, pred_svc2)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0,1], [0,1], 'k--' )
plt.rcParams['font.size'] = 12
plt.title('ROC curve for SVC')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.show()

# compute ROC AUC

from sklearn.metrics import roc_auc_score

ROC_AUC = roc_auc_score(validation_labels, pred_svc2)
print('ROC AUC : {:.4f}'.format(ROC_AUC))

#%% test-svc
test_svc2 = svc2.predict(tfidf_t)
