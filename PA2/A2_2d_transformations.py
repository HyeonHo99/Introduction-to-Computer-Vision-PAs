import os
import cv2
import numpy as np
from utils import *


def get_transformed_image(img, M):
    plane = np.ones((801,801), dtype=type(img[0][0]))

    coords = np.indices((img.shape[1], img.shape[0])).reshape(2,-1)
    coords[0] = img.shape[1]//2 - coords[0]
    coords[1] = img.shape[0]//2 - coords[1]
    coords = np.vstack((coords, np.ones(coords.shape[1])))
    before_x, before_y = coords[0], coords[1]

    # apply transformation M
    transformed_coords = np.round(M @ coords).astype(int) + 400
    after_x, after_y = transformed_coords[0], transformed_coords[1]

    plane[after_y, after_x] = img[(before_y+img.shape[0]//2).astype(int), (before_x+img.shape[1]//2).astype(int)]

    cv2.arrowedLine(plane, (400,0), (400,800), color=0, thickness=1, tipLength=0.01)
    cv2.arrowedLine(plane, (0,400), (800,400), color=0, thickness=1, tipLength=0.01)

    return plane


if __name__ == "__main__":
    dir = "CV_Assignment_2_Images/smile.png"
    smile = normalize(cv2.imread(dir,cv2.IMREAD_GRAYSCALE))

    if not os.path.exists("result"):
        os.makedirs("result")

    """
    Assume that the input image is odd sized along both dimensions.
    """
    print("1-2")

    # initial matrix M
    M = np.identity(3)
    cv2.imshow("smile", denormalize(get_transformed_image(smile, M)))

    # loop until 'q' is entered
    while(True):
        keyboard = cv2.waitKey()
        if keyboard == ord('q'):
            cv2.destroyAllWindows()
            break
        else:
            M = keyboard2matrix(keyboard, M)
            cv2.imshow("smile", denormalize(get_transformed_image(smile, M)))

