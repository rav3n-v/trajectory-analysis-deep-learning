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

#Checking correct values
correct=0
total=0
predicted_label=[]
actual_label=[]
#predictiong values for the test input
Predictions=model.predict(Combined_data_test)

#Categorizing the output
S={}


for i in range(70):
    S[i]=[]

#iterating thru predictions

for i in range(len(Predictions)):
    total += 1
    #converting predictions to labels
    value=np.argmax(Predictions[i])
    predicted_label.append(encoded[value])

    actual_label.append(encoded[target_test[i]])
    #checking Actual value
    if value==target_test[i]:
        correct+=1
    #Dviding the output into S1,S2,..S70, dividing the output based on the orginal targets
    S[target_test[i]].append(Predictions[i])


#Testing Model:
test_loss,test_accuracy = model.evaluate(Combined_data_test,target_test_vector,batch_size=1)

print(test_accuracy,test_loss)
print(correct/total)
print(matthews_corrcoef(actual_label,predicted_label))

#Printing Classification Report
print(classification_report(actual_label,predicted_label))

#T2 complexity calculation
M=[]
R=[]
T2=[]
#S is dictionary with the output as values(in a list form) and integer targets as keys
for i in S.keys():
    X = S[i]
    Mls=0
    maxnorm=0
    minnorm=X[0]
    #converting the targets from numerical to vectors
    g_Xk = np.array(tf.keras.utils.to_categorical(np.array(i), num_classes=70))

    for f_Xk in X:

        if np.linalg.norm(f_Xk-g_Xk)>Mls:
            Mls=np.linalg.norm(f_Xk-g_Xk)
        if np.linalg.norm(f_Xk)>np.linalg.norm(maxnorm):
            maxnorm=f_Xk
        if np.linalg.norm(f_Xk)<np.linalg.norm(minnorm):
            minnorm=f_Xk

    M.append(Mls)
    r=np.linalg.norm(maxnorm-minnorm)
    R.append(r)
    T2.append(np.abs(1-(Mls/r)))


print('T2 complexity',max(T2))

for i in range(len(T2)):
    print(encoded[i],':',T2[i])

