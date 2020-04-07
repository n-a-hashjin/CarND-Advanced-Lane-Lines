import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import time
#start_time = time.time()
#print("--- %s seconds ---" % (time.time() - start_time))

def cam_calibration(images_path, chess_size, show=False):
    chess_xsize = chess_size[0]
    chess_ysize = chess_size[1]

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chess_xsize*chess_ysize,3), np.float32)
    objp[:,:2] = np.mgrid[0:chess_xsize,0:chess_ysize].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane

    # Make a list of calibration images
    images = glob.glob(images_path)

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chess_size, None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            if show == 'all':
                #Draw and display the corners
                img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
                cv2.imshow('img',img)
                cv2.waitKey(500)

    if show != False:
        cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    ### TODO: src and dst should be define
    src = np.float32([[619,433],[260,690],[1060,690],[664,433]])
    dst = np.float32([[260,0],[260,720],[1060,720],[1060,0]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    camera_parameters = {"ret": ret, "mtx": mtx, "dist": dist, "rvecs": rvecs, "tvecs": tvecs, "M": M, "Minv": Minv}
    return camera_parameters

images_path = 'camera_cal/calibration*.jpg'
camera_parameters = cam_calibration(images_path, (9, 6))

def binary_image(image, mode="saturation", s_thresh=(90, 255),sx_thresh=(20,100)):
    if mode == "saturation":
        # 1) Convert to HLS color space
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]
        # 2) Apply a threshold to the S channel
        binary_output = np.zeros_like(s_channel)
        binary_output[(s_channel > s_thresh[0])&(s_channel <= s_thresh[1])] = 1
        # 3) Return a binary image of threshold result
    elif mode == "grad_saturation":
        #image = np.copy(image)
        # Convert to HLS color space and separate the V channel
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
        # Sobel x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        # Stack each channel
        #color_output = np.dstack((s_binary, sxbinary, np.zeros_like(sxbinary)))
        binary_output = s_binary | sxbinary
    return binary_output


def bird_eye(image, camera_parameters):
    mtx = camera_parameters["mtx"]
    dist = camera_parameters["dist"]
    dst = cv2.undistort(image, mtx, dist, None, mtx)
    M = camera_parameters["M"]
    warped = cv2.warpPerspective(dst, M, (dst.shape[1],dst.shape[0]), flags=cv2.INTER_LINEAR)
    return warped
"""
start_time = time.time()
for i in range(5000):
    test_img = cv2.imread("test_images/straight_lines2.jpg")
    bin_img = binary_image(test_img, mode="grad_saturation", s_thresh=(170, 255), sx_thresh=(30,100))
    warped = bird_eye(bin_img, camera_parameters)
ex_time = (time.time() - start_time)/(i+1)
print("--- %s seconds ---" % ex_time)
print(i)###->faster, 0.061 sec

start_time = time.time()
for j in range(5000):
    test_img = cv2.imread("test_images/straight_lines2.jpg")
    warped = bird_eye(test_img, camera_parameters)
    bin_img = binary_image(warped, mode="grad_saturation", s_thresh=(170, 255), sx_thresh=(30,100))
ex_time = (time.time() - start_time)/(j+1)
print("--- %s seconds ---" % ex_time)
print(i)###->slower, 0.074 sec

#cv2.imshow('top view',bin_img*255)
#cv2.waitKey()
#cv2.imwrite("output_images/bird_view.jpg", bin_img)
"""
test_img = cv2.imread("test_images/straight_lines2.jpg")
bin_img = binary_image(test_img, mode="grad_saturation", s_thresh=(170, 255), sx_thresh=(30,100))
warped = bird_eye(bin_img, camera_parameters)
v2.imshow('top view',bin_img*255)
cv2.waitKey()
cv2.imwrite("output_images/bird_view.jpg", bin_img)
