import os
import sys

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

from sklearn.metrics import confusion_matrix,classification_report

import numpy as np
import pandas as pd
from Functions import SBD_get_values_in
from sklearn.metrics import matthews_corrcoef



#Model
model = tf.keras.models.load_model('Model2.keras')

#Getting inputs
SBD_values_bd_test,SBD_values_fd_test,target_test,encoded=SBD_get_values_in((17,18,19,20))
input1_test = np.array(SBD_values_bd_test)
input2_test = np.array(SBD_values_fd_test)
#preparing data
Combined_data_test=np.concatenate((input1_test,input2_test),axis=1)
#preparing targets
target_test_vector = np.array(tf.keras.utils.to_categorical(np.array(target_test), num_classes=70))

Predictions=model.predict(Combined_data_test)

#Categorizing the output
S={}
cosine_similarity={}
cosine_similarity_2={}
for i in range(70):
    S[i]=[]
for i in range(len(Predictions)):
    S[target_test[i]].append(Predictions[i])

    cosine_similarity[target_test[i]]=[]
    cosine_similarity_2[target_test[i]]=[]

for i in S.keys():
    for k in S[i]:
        norm_k = np.linalg.norm(k)
        for j in range(70):
            if i!=j:
                    for m in S[j]:
                            cosine_similarity[i].append(np.dot(k,m)/(norm_k*np.linalg.norm(m)))


for i in S.keys():
    for j in S[i]:
        norm_j=np.linalg.norm(j)
        for m in S[i]:
            if (j!=m).all():
                cosine_similarity_2[i].append(np.dot(j,m)/(norm_j*np.linalg.norm(m)))


for i in cosine_similarity.keys():
    print(i,':',np.average(np.array(cosine_similarity[i])))
print('Cosine Similarity')
for i in cosine_similarity_2.keys():
    print(i,':',np.average(np.array(cosine_similarity_2[i])))
print(np.size(np.array(cosine_similarity[i])))



