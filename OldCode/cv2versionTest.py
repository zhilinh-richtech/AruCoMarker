import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

# Generate the marker
marker_id = 42
marker_size = 200  # marker area only
marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

# Add white border around the marker to simulate a real printed marker
border_size = 50  # pixels
bordered_image = cv2.copyMakeBorder(
    marker_image, border_size, border_size, border_size, border_size,
    cv2.BORDER_CONSTANT, value=255
)

# Save and show
cv2.imwrite('marker_42_bordered.png', bordered_image)
plt.imshow(bordered_image, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.title(f'ArUco Marker {marker_id} with Border')
plt.show()
