import cv2
import numpy as np

def process_img(img):
    """ Process img """
    alpha = 2 # Contrast control (1.0-3.0)
    beta = 0 # Brightness control (0-100)
    # blur = cv2.GaussianBlur(img, (1,1),0)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 49, 35)
    kernel = np.ones((5,5),np.uint8)
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("img", img)
    cv2.waitKey(0)


img = cv2.imread("../../TestInference/my_examples/test10.jpg", cv2.IMREAD_GRAYSCALE)
process_img(img)