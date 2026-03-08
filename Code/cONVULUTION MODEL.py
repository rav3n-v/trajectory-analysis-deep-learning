import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import cv2
import os
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix,classification_report
folder_path = 'C:/Users/vinay/PycharmProjects/Math Sem 6 Project/New'

all_files = os.listdir(folder_path)


with open('encoded.pickle', 'rb') as f:
    encoded = pickle.load(f)
# Filter PNG images using list comprehension

class CustomAboveOneInitializer(tf.keras.initializers.Initializer):
    def __init__(self, min_value=1.01, max_value=2.0):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, shape, dtype=None):
        return tf.random.uniform(shape, minval=self.min_value, maxval=self.max_value, dtype=dtype)

    def get_config(self):
        return {"min_value": self.min_value, "max_value": self.max_value}

# Example usage of custom initializer
initializer_1 = CustomAboveOneInitializer(min_value=-0.8, max_value=0.8)
initializer_2 = CustomAboveOneInitializer(min_value=-1/128, max_value=1/128)

png_images = [file for file in all_files if file.lower().endswith('.png')]
train=[]
test=[]
val=[]
train_data=[]
test_data=[]
val_data=[]
for i in png_images:
    print(i.split("-")[1].split(".")[0])
    if int(i.split("-")[1].split(".")[0]) in (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15):
        print(i)
        name=i.split("-")[0]
        train_data.append(cv2.imread('New/' + i))

        train.append(list(encoded.keys())[list(encoded.values()).index(name)])

    if int(i.split("-")[1].split(".")[0]) in (15,16):
        name=i.split("-")[0]
        val_data.append(cv2.imread('New/' + i))
        val.append(list(encoded.keys())[list(encoded.values()).index(name)])
    if int(i.split("-")[1].split(".")[0])in (17,18,19,20):
        test_data.append( cv2.imread('New/' + i))
        name=i.split("-")[0]
        test.append(list(encoded.keys())[list(encoded.values()).index(name)])



target_train = np.array(tf.keras.utils.to_categorical(np.array(train), num_classes=70))
target_val = np.array(tf.keras.utils.to_categorical(np.array(val), num_classes=70))
target_test = np.array(tf.keras.utils.to_categorical(np.array(test), num_classes=70))


#Variables
INPUT_SHAPE = (128, 128, 3)
FILTER1_SIZE = 32
FILTER2_SIZE = 64
FILTER_SHAPE = (3, 3)
POOL_SHAPE = (2, 2)
FULLY_CONNECT_NUM = 128
NUM_CLASSES = 70

# Model architecture implementation



model = Sequential([
    layers.Rescaling(1. / 255, input_shape=(128, 128, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='tanh',kernel_initializer=initializer_2),
    layers.Dense(NUM_CLASSES,activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Use sparse categorical crossentropy for integer targets
              metrics=['categorical_accuracy'])


model.fit(np.array(train_data),target_train,epochs=5, batch_size=1)

#predictiong values for the test input
Predictions=model.predict(np.array(test_data))
total=0
correct=0
predicted_label=[]
actual_label=[]
for i in range(len(Predictions)):
    total += 1
    #converting predictions to labels
    value=np.argmax(Predictions[i])
    predicted_label.append(encoded[value])

    actual_label.append(encoded[test[i]])
    #checking Actual value
    if value==test[i]:
        correct+=1


#Testing Model:
test_loss,test_accuracy = model.evaluate(np.array(test_data),target_test,batch_size=1)

print(test_accuracy,test_loss)
print(correct/total)


#Printing Classification Report
print(classification_report(actual_label,predicted_label))

