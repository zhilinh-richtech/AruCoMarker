import cv2 as cv
cols = 5
rows = 7
square_size = 100
marker_size = 0.8*square_size
aruco_dict = cv.aruco.getPredefinedDictionary(0) # cv.aruco.DICT_4X4_250
board = cv.aruco.CharucoBoard((cols, rows), square_size, marker_size, aruco_dict)
# board.setLegacyPattern(True) # comment this line to create the new template
from_cv_img = board.generateImage((cols*square_size, rows*square_size))
cv.imshow("board_6_4_dict_0", from_cv_img)
cv.imwrite("board_6_4_dict_0.png", from_cv_img)