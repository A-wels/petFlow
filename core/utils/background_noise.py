import matplotlib.image
import cv2
import numpy as np


# add noise to the background of the image
def add_noise_to_background(frame):
    frame = frame.T
    # save image before
    matplotlib.image.imsave("viz_results/before.png", frame)
    
    # dilate image to increase size of blob
    kernel = np.ones((9,9),np.uint8) 
    frame_dilated = cv2.dilate(frame, kernel, iterations=1)

    # find blob in center with opencv
    ret, thresh = cv2.threshold(frame_dilated, 4, 255, 0)
    # save thresh image
    matplotlib.image.imsave("viz_results/thresh.png", thresh)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #get the largest contour
    cnt = max(contours, key=cv2.contourArea)
    # make contour a convex hull
    cnt = cv2.convexHull(cnt)

    # draw convex hull in copy of image and save
    frame_copy = frame.copy()
    cv2.drawContours(frame_copy, [cnt], 0, (255, 255, 255), -1)
    matplotlib.image.imsave("viz_results/convex_hull.png", frame_copy)


    # remove area outside of convex hull
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            if cv2.pointPolygonTest(cnt, (j, i), False) < 0:
                frame[i, j] = 0
            else:
                frame[i, j] = frame[i, j]
    # convert to grayscale

    matplotlib.image.imsave("viz_results/after.png", frame)
    exit()