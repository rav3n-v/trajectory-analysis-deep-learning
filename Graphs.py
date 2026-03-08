import numpy as np
import cv2
import os
folder_path_1 = 'C:/Users/vinay/OneDrive/Desktop/Project/Code/images/PNG'
folder_path_2 = 'C:/Users/vinay/PycharmProjects/Math Sem 6 Project/New'



all_files = os.listdir(folder_path_1)
png_images = [file for file in all_files if file.lower().endswith('.png')]
for i in png_images:
    image = cv2.imread(folder_path_1+'/'+i)
    image=cv2.resize(image,(128,128))
    os.chdir(folder_path_2)
    cv2.imwrite(i,image)

