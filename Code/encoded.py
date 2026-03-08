import pickle
from Functions import SBD_get_values_in
import numpy as np
d={}


for i in range(1,21):
    SBD_values_bd, SBD_values_fd, target, encoded = SBD_get_values_in([i])
    input1 = np.array(SBD_values_bd)
    input2 = np.array(SBD_values_fd)
    Combined_data = np.concatenate((input1, input2), axis=1)
    d[i]=Combined_data
print(d)
with open('sbd_values.pickle', 'wb')as f:
    pickle.dump(d,f)