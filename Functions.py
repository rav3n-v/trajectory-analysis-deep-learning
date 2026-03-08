import cv2
import os
import numpy as np
import pickle
import tensorflow as tf
folder_path = 'C:/Users/vinay/OneDrive/Desktop/Project/Project 1/Code/images/PNG'

def read_images(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    apple_trajectory = largest_contour.squeeze().tolist()
    apple_trajectory = np.array(apple_trajectory)
    # Calculate cumulative distance between consecutive points
    distances = np.sqrt(np.sum(np.diff(apple_trajectory, axis=0) ** 2, axis=1))
    cumulative_distances = np.concatenate(([0], np.cumsum(distances)))
    total_length = cumulative_distances[-1]
    num_points = 100
    spacing = total_length / (num_points - 1)
    # Resample the curve
    new_points = []
    for i in range(num_points):
        distance_along_curve = i * spacing
        # Find the index corresponding to the distance along the curve
        idx = np.searchsorted(cumulative_distances, distance_along_curve)

        # Interpolate to find the point at the specified distance
        if idx == 0:
            new_point = apple_trajectory[0]
        elif idx == len(apple_trajectory):
            new_point = apple_trajectory[-1]
        else:
            t = (distance_along_curve - cumulative_distances[idx - 1]) / (
                        cumulative_distances[idx] - cumulative_distances[idx - 1])
            new_point = (1 - t) * apple_trajectory[idx - 1] + t * apple_trajectory[idx]
        new_points.append(new_point)

    new_points = np.array(new_points)
    resampled_points = new_points
    q1 = resampled_points[0]
    # List to store triangles
    triangles = []

    # Iterate over the rest of the points to form triangles
    for i in range(1, len(resampled_points) - 1):
        q2 = resampled_points[i]
        q3 = resampled_points[i + 1]
        # Add the triangle to the list
        triangles.append((q1, q2, q3))
    triangles = np.array(triangles)

    def calculate_barycenter(triangle_coords):
        x_bc_sum = 0
        y_bc_sum = 0
        total_area = 0

        for triangle in triangle_coords:
            # Extract coordinates of triangle vertices
            x_q1, y_q1 = triangle[0]
            x_q2, y_q2 = triangle[1]
            x_q3, y_q3 = triangle[2]

            # Calculate area of the triangle using shoelace formula
            area = 0.5 * abs(x_q1 * (y_q2 - y_q3) + x_q2 * (y_q3 - y_q1) + x_q3 * (y_q1 - y_q2))
            total_area += area

            # Calculate centroid of the triangle and add to sum
            x_bc_sum += area * (x_q1 + x_q2 + x_q3) / 3
            y_bc_sum += area * (y_q1 + y_q2 + y_q3) / 3

        x_bc = x_bc_sum / total_area
        y_bc = y_bc_sum / total_area

        return x_bc, y_bc, total_area

    # Calculate the barycenter
    xbc, ybc, total_area = calculate_barycenter(triangles)

    for i in range(len(resampled_points)):
        resampled_points[i][0] = (resampled_points[i][0] - xbc) / total_area
        resampled_points[i][1] = (resampled_points[i][1] - ybc) / total_area

    SBD = []

    def angle(v1, v2):
        # Convert vectors to numpy arrays
        v1_np = np.array(v1)
        v2_np = np.array(v2)

        # Calculate the cross product
        cross_product = np.cross(v1_np, v2_np)

        # Calculate the dot product
        dot_product = np.dot(v1_np, v2_np)

        # Calculate the angle in radians
        angle_rad = np.arctan2(cross_product, dot_product)

        # Ensure the angle is between 0 and 2*pi
        angle_rad %= 2 * np.pi

        return 2 * np.pi - angle_rad

    distances_between_points = np.sqrt(np.sum(np.diff(resampled_points, axis=0) ** 2, axis=1))
    for i in range(len(resampled_points)):
        if i == 0:
            SBD.append((np.linalg.norm(resampled_points[0]), 0))
        else:
            angle2 = angle(resampled_points[i - 1], resampled_points[i] - resampled_points[i - 1])
            if 0 <= angle2 < np.pi:
                SBD.append((-np.linalg.norm(resampled_points[i]), distances_between_points[i - 1]))
            else:
                SBD.append((np.linalg.norm(resampled_points[i]), distances_between_points[i - 1]))
    SBD[0] = (SBD[0][0] * SBD[1][0] / np.abs(SBD[1][0]), 0)
    countp = 0
    countn = 0

    return np.array(SBD)
def DSBD(SBD1,SBD2,m=100):
    D=0
    for i in range(m):
        if (SBD1[i][0]> 0 and SBD2[i][0]>0) or(SBD1[i][0]< 0 and SBD2[i][0]<0):
            w=1
        else:
            w=2
        D+=2*w*np.abs(np.abs(SBD1[i][0])-np.abs(SBD2[i][0]))/(np.abs(SBD1[i][0])+np.abs(SBD2[i][0]))
    return D/m
def SBD():
    # Get all files in the folder
    global folder_path
    all_files = os.listdir(folder_path)

    # Filter PNG images using list comprehension
    png_images = [file for file in all_files if file.lower().endswith('.png')]

    SBD_values_bd = []
    SBD_values_fd = []
    count = -1
    input_size = 0
    encoded = {}
    target = []
    for i in png_images:

        path_image = folder_path + '/' + i
        SBD = read_images(path_image)
        name = i.split('-')[0]
        if name in encoded.values():
            if input_size < 15:
                target.append(count)
                SBD_values_bd.append(SBD[:, 0])
                SBD_values_fd.append(SBD[:, 1])
                input_size += 1
        else:
            count += 1
            target.append(count)
            SBD_values_bd.append(SBD[:, 0])
            SBD_values_fd.append(SBD[:, 1])
            encoded[count] = name

            input_size = 0

    SBD_values_bd = np.array(SBD_values_bd)
    SBD_values_fd = np.array(SBD_values_fd)
    return SBD_values_bd,SBD_values_fd
def SBD_get_values():
    # Define the path to the folder containing the images
    global folder_path

    # Get all files in the folder
    all_files = os.listdir(folder_path)

    # Filter PNG images using list comprehension
    png_images = [file for file in all_files if file.lower().endswith('.png')]

    SBD_values_bd = []
    SBD_values_fd = []
    count = -1
    input_size = 0
    encoded = {}

    target = []
    for i in png_images:

        path_image = folder_path + '/' + i
        SBD = read_images(path_image)
        name = i.split('-')[0]
        if name in encoded.values():
            if input_size < 15:
                target.append(count)
                SBD_values_bd.append(SBD[:, 0])
                SBD_values_fd.append(SBD[:, 1])
                input_size += 1
        else:
            count += 1
            target.append(count)
            SBD_values_bd.append(SBD[:, 0])
            SBD_values_fd.append(SBD[:, 1])
            encoded[count] = name

            input_size = 0

    SBD_values_bd = np.array(SBD_values_bd)
    SBD_values_fd = np.array(SBD_values_fd)
    return SBD_values_bd,SBD_values_fd,target,encoded


def SBD_bd_fd(path_image):
    SBD_values_bd=[]
    SBD_values_fd=[]

    SBD = read_images(path_image)
    SBD_values_bd.append(SBD[:, 0])
    SBD_values_fd.append(SBD[:, 1])

    input1 = np.array(SBD_values_bd)
    input2 = np.array(SBD_values_fd)

    Combined_data = np.concatenate((input1, input2), axis=1)

    return Combined_data

#SBD_values_bd,SBD_values_fd,target,encoded=SBD_get_values()

#with open("all_values.pickle", "wb") as f:
    #pickle.dump({"SBD_values_bd":SBD_values_bd,"SBD_values_fd":SBD_values_fd,'target':target,'encoded':encoded}, f)
def SBD_get_values_in(a):
    # Define the path to the folder containing the images
    global folder_path

    # Get all files in the folder
    all_files = os.listdir(folder_path)

    # Filter PNG images using list comprehension
    png_images = [file for file in all_files if file.lower().endswith('.png')]

    SBD_values_bd = []
    SBD_values_fd = []

    with open('encoded.pickle','rb') as f:
        encoded = pickle.load(f)

    target = []
    for i in png_images:

        path_image = folder_path + '/' + i
        SBD = read_images(path_image)
        name = i.split('-')[0]
        num=int(i.split('-')[1].split('.')[0])

        if num in a:
                target.append(list(encoded.keys())[list(encoded.values()).index(name)])
                SBD_values_bd.append(SBD[:, 0])
                SBD_values_fd.append(SBD[:, 1])







    SBD_values_bd = np.array(SBD_values_bd)
    SBD_values_fd = np.array(SBD_values_fd)
    return SBD_values_bd,SBD_values_fd,target,encoded
def encoded():
    # Define the path to the folder containing the images
    global folder_path


    # Get all files in the folder
    all_files = os.listdir(folder_path)

    # Filter PNG images using list comprehension
    png_images = [file for file in all_files if file.lower().endswith('.png')]
    encoded={}
    count=0
    for i in png_images:
        name=i.split("-")[0]
        if name not in encoded.values():

            encoded[count]=name
            count += 1
    print(encoded)
    return encoded


#encoded=encoded()
#with open('encoded.pickle','wb')as f:
    #pickle.dump(encoded,f)
def SBD_VALUES_3(A):
    # Data
    with open('sbd_values.pickle', 'rb') as f:
        SBD_VALUES = pickle.load(f)
    input=np.array([])
    target = []
    for i in A:
        np.concatenate(input,SBD_VALUES[i])
        target += list(range(70))
    return np.array(input),np.array(target)



def SBD_get_values_in_normalised(a):
    # Define the path to the folder containing the images
    global folder_path

    # Get all files in the folder
    all_files = os.listdir(folder_path)

    # Filter PNG images using list comprehension
    png_images = [file for file in all_files if file.lower().endswith('.png')]

    SBD_values_bd = []
    SBD_values_fd = []

    with open('encoded.pickle','rb') as f:
        encoded = pickle.load(f)

    target = []
    for i in png_images:

        path_image = folder_path + '/' + i
        SBD = read_images(path_image)
        name = i.split('-')[0]
        num=int(i.split('-')[1].split('.')[0])

        if num in a:
                target.append(list(encoded.keys())[list(encoded.values()).index(name)])
                SBD_values_bd.append(SBD[:, 0])
                SBD_values_fd.append(SBD[:, 1])







    SBD_values_bd = np.array( )
    SBD_values_fd = np.array(SBD_values_fd)
    SBD_values_bd = tf.keras.layers.Normalization(axis=-1)

    return SBD_values_bd,SBD_values_fd,target,encoded
