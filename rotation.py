import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the black and white image
image = cv2.imread('C:/Users/vinay/OneDrive/Desktop/Project/Code/images/PNG/spring-11.png', cv2.IMREAD_GRAYSCALE)

# Apply thresholding
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Extract the largest contour (assuming it represents the apple)
largest_contour = max(contours, key=cv2.contourArea)

# Extract coordinates of the contour points
apple_trajectory = largest_contour.squeeze().tolist()

# Ploting The coordinates
x_coords = [point[0] for point in apple_trajectory]
y_coords = [point[1] for point in apple_trajectory]
plt.scatter(x_coords, y_coords)

plt.show()
apple_trajectory = np.array(apple_trajectory)
# Calculate cumulative distance between consecutive points
distances = np.sqrt(np.sum(np.diff(apple_trajectory, axis=0) ** 2, axis=1))
cumulative_distances = np.concatenate(([0], np.cumsum(distances)))

# Determine total length of the curve
total_length = cumulative_distances[-1]

# Define the number of points you want to resample
num_points = 100

# Calculate spacing between points
spacing = total_length / (num_points - 1)

# Resample the curve
resampled_points = []
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
    resampled_points.append(new_point)

# Plot resampled points
resampled_points = np.array(resampled_points)
plt.plot(resampled_points[:, 0], resampled_points[:, 1], 'ro-', label='Resampled Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Resampled Apple Trajectory')
plt.show()

# List to store triangles
triangles = []

# Iterate over the rest of the points to form triangles
q1 = resampled_points[0]
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

        # Calculate area of the triangle
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
print("Barycenter (xbc, ybc):", (xbc, ybc))
plt.plot(resampled_points[:, 0], resampled_points[:, 1], 'ro-', label='Resampled Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(xbc, ybc, color='b')
plt.show()
# Setting Barrycenter as the orgin
for i in range(len(resampled_points)):
    resampled_points[i][0] = (resampled_points[i][0] - xbc) / total_area
    resampled_points[i][1] = (resampled_points[i][1] - ybc) / total_area

print("Barycenter (xbc, ybc):", (xbc, ybc))
plt.plot(resampled_points[:, 0], resampled_points[:, 1], 'ro-', label='Resampled Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Assuming resampled_points is your array of points
# Extract x and y coordinates separately
x_coords = resampled_points[:, 0]
y_coords = resampled_points[:, 1]

# Initialize lists to store norms for x and y axes
norm_x = []
norm_y = []
norm_BD = []
# Calculate norm for x and y axes separately for each point
for i in range(len(resampled_points)):
    norm_x.append(np.linalg.norm(x_coords[i]))
    norm_y.append(np.linalg.norm(y_coords[i]))
    norm_BD.append(np.linalg.norm(resampled_points[i]))
    # min of BD
    min_value = min(norm_BD)
# Assuming BD is your array
min_index = np.argmin(norm_BD)
print("Index of minimum value:", min_index)
# Index 89 element from norm_x and norm_y arrays
element_norm_x = norm_x[min_index]
element_norm_y = norm_y[min_index]

print("Element at min index  from norm_x:", element_norm_x)
print("Element at min index  from norm_y:", element_norm_y)

# Define vectors
v = np.array([0, 0.001])  # Vector (0, 0.001)
w = np.array([element_norm_x, element_norm_y])  # Vector formed by x and y axes

# Calculate the cross product
cross_product = np.cross(v, w)

# Calculate the dot product
dot_product = np.dot(v, w)

# Calculate the angle in radians using the arctan2 function to handle all cases
angle_rad = np.arctan2(cross_product, dot_product)

# Ensure the angle is between 0 and 2*pi
angle_rad %= 2 * np.pi

# Convert angle to the range [0, 2*pi)
angle_rad = 2 * np.pi - angle_rad

print("Angle between vectors (in radians):", angle_rad)

xy_matrix = np.column_stack((x_coords, y_coords))
# Display the matrix
print(xy_matrix)
# the rotation matrix


# Assuming theta is the angle in radians
theta = angle_rad

# Calculate cos(theta) and sin(theta)
cos_theta = np.cos(theta)
sin_theta = np.sin(theta)

# Create the 2x2 matrix
import numpy as np

# Calculate cos(theta) and sin(theta)
cos_theta = np.cos(theta)
sin_theta = np.sin(theta)

# Create the 2x2 matrix
rotation_matrix = np.array([[cos_theta, sin_theta],
                            [-sin_theta, cos_theta]])

print("Rotation matrix:")
print(rotation_matrix)
transformed_matrix = np.dot(xy_matrix, rotation_matrix)

print("Transformed matrix:")
print(transformed_matrix)
import matplotlib.pyplot as plt

# Assuming transformed_matrix is the transformed coordinates
# Extract x and y coordinates separately from the transformed matrix
x_coords_transformed = transformed_matrix[:, 0]
y_coords_transformed = transformed_matrix[:, 1]
transformed_points = np.array(list(zip(x_coords_transformed, y_coords_transformed)))
print(transformed_points)
# Plot the transformed coordinates as a scatter plot
plt.scatter(x_coords_transformed, y_coords_transformed, color='blue', label='Transformed Points')

# Label axes and add legend
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

# Show plot
plt.show()