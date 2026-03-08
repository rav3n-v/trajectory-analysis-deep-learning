
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

from Functions import SBD_get_values_in


class CustomAboveOneInitializer(tf.keras.initializers.Initializer):
    def __init__(self, min_value=1.01, max_value=2.0):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, shape, dtype=None):
        return tf.random.uniform(shape, minval=self.min_value, maxval=self.max_value, dtype=dtype)

    def get_config(self):
        return {"min_value": self.min_value, "max_value": self.max_value}

# Example usage of custom initializer
initializer_1 = CustomAboveOneInitializer(min_value=-5, max_value=5)
initializer_2 = CustomAboveOneInitializer(min_value=-1/128, max_value=1/128)
initializer_3 = CustomAboveOneInitializer(min_value=-1/100, max_value=1/100)



SBD_values_bd,SBD_values_fd,target,encoded=SBD_get_values_in((1,2,3,4,5,6,7,8,9,10,11,12))
SBD_values_bd_test,SBD_values_fd_test,target_test,encoded=SBD_get_values_in((17,18,19,20))


input1 = np.array(SBD_values_bd)
input2 = np.array(SBD_values_fd)
input1_test = np.array(SBD_values_bd_test)
input2_test = np.array(SBD_values_fd_test)
#preparing data
Combined_data_test=np.concatenate((input1_test,input2_test),axis=1)
#preparing targets
target_test_vector = np.array(tf.keras.utils.to_categorical(np.array(target_test), num_classes=70))

Combined_data=np.concatenate((input1,input2),axis=1)

target = np.array(target)
target1 = tf.keras.utils.to_categorical(target, num_classes=70)  # Assuming there are 70 categories
target1 = np.array(target1)



num_classes=70

model = tf.keras.Sequential([
tf.keras.layers.Dense(128, activation='tanh', input_shape=(200,),kernel_initializer=initializer_2),
tf.keras.layers.Dense(100, activation='tanh', input_shape=(200,),kernel_initializer=initializer_3),

tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='MSE'
              ,  # Use sparse categorical crossentropy for integer targets
              metrics=['categorical_accuracy'])

model.fit(Combined_data, target1, epochs=100, batch_size=1)
#mODEL_2
model_2 = tf.keras.Sequential([
tf.keras.layers.Dense(128, activation='tanh', input_shape=(200,),kernel_initializer=initializer_1),
tf.keras.layers.Dense(100, activation='tanh', input_shape=(200,),kernel_initializer=initializer_1),

    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model_2.compile(optimizer='adam',
              loss='MSE'
              ,  # Use sparse categorical crossentropy for integer targets
              metrics=['categorical_accuracy'])

model_2.fit(Combined_data, target1, epochs=100, batch_size=1)

# Evaluate the model
test_loss, test_acc = model.evaluate(Combined_data_test, target_test_vector)

print('Test accuracy Model_1:', test_acc)
print(np.array([Combined_data[0]]).shape)
#MODEL _2 RESULTS
test_loss, test_acc = model_2.evaluate(Combined_data_test, target_test_vector)
model_2
print('Test accuracy Model_2:', test_acc)

v=model.predict(Combined_data_test)
v=np.argmax(v,axis=1)
for i in v:
    print(encoded[i])
print(v.shape)
model.save('Model.keras')
model_2.save('Model_2.keras')

