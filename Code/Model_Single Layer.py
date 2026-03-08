#Import Statements
import pickle
import numpy as np
import os
import seaborn as sns
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import matplotlib.pyplot as plt
from Functions import SBD_get_values_in
sns.set()
sns.set_theme(style="ticks", palette="pastel")
model1 = tf.keras.models.load_model('Model2.keras')
from random import shuffle
import sys
class CustomAboveOneInitializer(tf.keras.initializers.Initializer):
    def __init__(self, min_value=1.01, max_value=2.0):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, shape, dtype=None):
        return tf.random.uniform(shape, minval=self.min_value, maxval=self.max_value, dtype=dtype)

    def get_config(self):
        return {"min_value": self.min_value, "max_value": self.max_value}



#Getting Inputs and Encoded Dictionary
L=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
test_accuracies=[]
Default_Accuracy=[]

shuffle(L)
SBD_values_bd, SBD_values_fd, target, encoded = SBD_get_values_in(L[0:16])
SBD_values_bd_val_test, SBD_values_fd_val_test, target_val_test, encoded_val_test = SBD_get_values_in(L[14:16])
SBD_values_bd_test, SBD_values_fd_test, target_test, encoded_test = SBD_get_values_in(L[16:20])

# Rearranging Inputs
input1 = np.array(SBD_values_bd)
input2 = np.array(SBD_values_fd)
input1_val_test = np.array(SBD_values_bd_val_test)
input2_val_test = np.array(SBD_values_fd_val_test)
input1_test = np.array(SBD_values_bd_test)
input2_test = np.array(SBD_values_fd_test)

Combined_data = np.concatenate((input1, input2), axis=1)
Combined_data_val_test = np.concatenate((input1_val_test, input2_val_test), axis=1)
Combined_data_test = np.concatenate((input1_test, input2_test), axis=1)

# Converting targets to vectors of 70 dimension
target = np.array(
    tf.keras.utils.to_categorical(np.array(target), num_classes=70))  # Assuming there are 70 categories
target_val_test = np.array(tf.keras.utils.to_categorical(np.array(target_val_test), num_classes=70))
target_test = np.array(tf.keras.utils.to_categorical(np.array(target_test), num_classes=70))
# Model Without varying initialisation
num_classes = 70
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='tanh', input_shape=(200,)),

    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Use sparse categorical crossentropy for integer targets
              metrics=['categorical_accuracy'])

training_history = model.fit(Combined_data, target, epochs=100, batch_size=1, verbose=0)
print('Model 3', 'Default')
test_loss_3, test_accuracy_3 = model.evaluate(Combined_data_test, target_test, batch_size=1)
print(test_accuracy_3, test_loss_3)
Default_Accuracy.append(test_accuracy_3)



for i in range(20):
    initializer_1 = CustomAboveOneInitializer(min_value=-i * 0.1, max_value=i * 0.1)
    #Model With varying initialisation
    num_classes=70
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128,activation='tanh',kernel_initializer=initializer_1,input_shape=(200,)),

        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',  # Use sparse categorical crossentropy for integer targets
                  metrics=['categorical_accuracy'])

    training_history=model.fit(Combined_data, target, epochs=100, batch_size=1,verbose=0)


    # Evaluate the model
    print('Model 3')
    test_loss_3,test_accuracy_3=model.evaluate(Combined_data_test,target_test,batch_size=1)
    print(i,':',test_accuracy_3,test_loss_3)

    test_accuracies.append(test_accuracy_3)


print('Results with varying initialisation')
for i in test_accuracies:
    print(i)
print('Results with default initialisation')
for i in Default_Accuracy:
    print(i)