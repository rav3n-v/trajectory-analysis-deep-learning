
#Import Statements

import pickle
import sys

import numpy as np
import os
import seaborn as sns
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from Functions import SBD_get_values_in
from tensorflow.keras.layers import Normalization
sns.set()
sns.set_theme(style="ticks", palette="pastel")

class CustomAboveOneInitializer(tf.keras.initializers.Initializer):
    def __init__(self, min_value=1.01, max_value=2.0):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, shape, dtype=None):
        return tf.random.uniform(shape, minval=self.min_value, maxval=self.max_value, dtype=dtype)

    def get_config(self):
        return {"min_value": self.min_value, "max_value": self.max_value}

# Example usage of custom initializer
initializer_1 = CustomAboveOneInitializer(min_value=-1/200, max_value=1/200)
initializer_2 = CustomAboveOneInitializer(min_value=-1/128, max_value=1/128)
# Example usage of custom initializer
initializer_3 = CustomAboveOneInitializer(min_value=-5, max_value=5)
initializer_4 = CustomAboveOneInitializer(min_value=-5, max_value=5)

#Getting Inputs and Encoded Dictionary

SBD_values_bd,SBD_values_fd,target,encoded=SBD_get_values_in((1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20))
for i in target:
    print(i)


sys.exit()
SBD_values_bd_val_test,SBD_values_fd_val_test,target_val_test,encoded_val_test=SBD_get_values_in((15,16))
SBD_values_bd_test,SBD_values_fd_test,target_test,encoded_test=SBD_get_values_in((17,18,19,20))
print(SBD_values_bd[0])
print(SBD_values_fd[0])
print('Apple 2')
print(SBD_values_bd[1])
print(SBD_values_fd[1])

#Rearranging Inputs
input1 = np.array(SBD_values_bd)
input2 = np.array(SBD_values_fd)
input1_val_test = np.array(SBD_values_bd_val_test)
input2_val_test = np.array(SBD_values_fd_val_test)
input1_test = np.array(SBD_values_bd_test)
input2_test = np.array(SBD_values_fd_test)

Combined_data=np.concatenate((input1,input2),axis=1)
Combined_data_val_test=np.concatenate((input1_val_test,input2_val_test),axis=1)
Combined_data_test=np.concatenate((input1_test,input2_test),axis=1)
norm_layer = Normalization()
norm_layer.adapt(Combined_data)



sys.exit()

# Apply normalization to data
Combined_data = norm_layer(Combined_data)
Combined_data_val_test = norm_layer(Combined_data_val_test)
Combined_data_test = norm_layer(Combined_data_test)
#Converting targets to vectors of 70 dimension
target = np.array(tf.keras.utils.to_categorical(np.array(target), num_classes=70)) # Assuming there are 70 categories
target_val_test = np.array(tf.keras.utils.to_categorical(np.array(target_val_test), num_classes=70))
target_test = np.array(tf.keras.utils.to_categorical(np.array(target_test), num_classes=70))

#Model 1
num_classes=70
model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(128,activation='tanh',input_shape=(200,),kernel_initializer=initializer_1),
    tf.keras.layers.Dense(100,activation='tanh',kernel_initializer=initializer_2),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model_1.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy',  # Use sparse categorical crossentropy for integer targets
              metrics=['categorical_accuracy'])

training_history=model_1.fit(Combined_data, target, epochs=100, batch_size=1)

#Model 2

num_classes=70
model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(128,activation='tanh',input_shape=(200,),kernel_initializer=initializer_3),
    tf.keras.layers.Dense(100,activation='tanh',kernel_initializer=initializer_4),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model_2.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Use sparse categorical crossentropy for integer targets
              metrics=['categorical_accuracy'])

training_history_2=model_2.fit(Combined_data, target, epochs=100, batch_size=1)

#validation_data=[Combined_data_val_test,target_val_test]
#Plotting
fig,(ax1,ax2)=plt.subplots(1,2)
fig.suptitle('Model_1 vs Model_2,Activation Function: Tanh')
plt.setp([ax2],xlim=(0,100),ylim=(0,1))
#Loss/Validation Loss
ax1.plot(training_history.history['loss'],'r',label='loss_Model_1')
ax1.plot(training_history_2.history['loss'],'b',label='loss_Model_2')
ax1.legend()
ax1.set(xlabel='Epochs',ylabel='Loss')


#Accuracy/Validation Accuracy
ax2.plot(training_history.history['categorical_accuracy'],'r',label='accuracy_Model_1')
ax2.plot(training_history_2.history['categorical_accuracy'],'b',label='accuracy_Model_2')
ax2.legend()
ax2.set(xlabel='Epochs',ylabel='Accuracy')
fig.tight_layout()

plt.show()

# Evaluate the model
print('Model 1')
test_loss,test_accuracy=model_1.evaluate(Combined_data_test,target_test,batch_size=1)
print(test_accuracy,test_loss)
print('Model 2')
test_loss,test_accuracy=model_2.evaluate(Combined_data_test,target_test,batch_size=1)
print(test_accuracy,test_loss)



