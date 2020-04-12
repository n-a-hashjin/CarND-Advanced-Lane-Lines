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
    src = np.float32([[619,434],[260,690],[1060,690],[664,434]])
    dst = np.float32([[260,0],[260,720],[1060,720],[1060,0]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    camera_parameters = {"ret": ret, "mtx": mtx, "dist": dist, "rvecs": rvecs, "tvecs": tvecs, "M": M, "Minv": Minv}
    return camera_parameters

images_path = 'camera_cal/calibration*.jpg'
camera_parameters = cam_calibration(images_path, (9, 6))

def binary_image(image, mode="grad_saturation", s_thresh=(90, 255),sx_thresh=(20,100)):
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
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        h_channel = hls[:,:,0]
        s_channel = hls[:,:,2]

        # Sobel x
        #gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(h_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
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
    elif mode == "hsv_filter":
        #blured_img = cv2.GaussianBlur(image, (5, 5), 0)
        lower_ylw = np.array([15,90,160])
        upper_ylw = np.array([27,255,255])
        lower_wht = np.array([0,0,200])
        upper_wht = np.array([180,25,255])

        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        #select yellow areas
        mask_ylw = cv2.inRange(hsv_img, lower_ylw, upper_ylw)
        #select white areas
        mask_wht = cv2.inRange(hsv_img, lower_wht, upper_wht)
        #combine to white and yellow masks together
        binary_output = cv2.bitwise_or(mask_ylw, mask_wht)
        binary_output[binary_output != 0] = 1
    return binary_output

def bird_eye(image, camera_parameters):
    mtx = camera_parameters["mtx"]
    dist = camera_parameters["dist"]
    dst = cv2.undistort(image, mtx, dist, None, mtx)
    M = camera_parameters["M"]
    warped = cv2.warpPerspective(dst, M, (dst.shape[1],dst.shape[0]), flags=cv2.INTER_LINEAR)
    return warped

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin



        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        print('The function failed to find indices!')

        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    return leftx, lefty, rightx, righty

def search_around_poly(binary_warped, left_poly, right_poly):
    # Choose the width of the margin around the previous polynomial to search
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = []
    right_lane_inds = []
    left_lane_inds = ((nonzerox > (left_poly[0]*(nonzeroy**2) + left_poly[1]*nonzeroy +
                    left_poly[2] - margin)) & (nonzerox < (left_poly[0]*(nonzeroy**2) +
                    left_poly[1]*nonzeroy + left_poly[2] + margin)))
    right_lane_inds = ((nonzerox > (right_poly[0]*(nonzeroy**2) + right_poly[1]*nonzeroy +
                    right_poly[2] - margin)) & (nonzerox < (right_poly[0]*(nonzeroy**2) +
                    right_poly[1]*nonzeroy + right_poly[2] + margin)))

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty

def find_path(binary_warped, left_poly, right_poly):
    try:
        leftx, lefty, rightx, righty = search_around_poly(binary_warped, left_poly, right_poly)
    except:
        print('search from scratch!!!!!')
        leftx, lefty, rightx, righty = find_lane_pixels(binary_warped)
        pass
    return leftx, lefty, rightx, righty

def fit_poly(img_shape, leftx, lefty, rightx, righty):
    # Fit a second order polynomial to each using `np.polyfit`
    try:
        left_poly = np.polyfit(lefty, leftx, 2)
        right_poly = np.polyfit(righty, rightx, 2)
    except:
        left_poly = []
        right_poly = []
    return left_poly, right_poly

def path_visualization(frame, Minv, left_line, right_line):
    left_poly = left_line.best_fit
    right_poly = right_line.best_fit
    leftx = left_line.allx
    lefty = left_line.ally
    rightx = right_line.allx
    righty = right_line.ally
    ploty = right_line.ploty


    left_fitx = left_poly[0]*ploty**2 + left_poly[1]*ploty + left_poly[2]
    right_fitx = right_poly[0]*ploty**2 + right_poly[1]*ploty + right_poly[2]

    road_poly = np.zeros_like(frame)
    left_side = np.transpose(np.vstack((left_fitx, ploty)))
    right_side = np.flipud(np.transpose(np.vstack((right_fitx, ploty))))
    road_poly_pts = np.array([np.vstack((left_side, right_side))])
    if (left_line.detected & right_line.detected) == True:
        cv2.fillPoly(road_poly, np.int_(road_poly_pts), (0, 255, 0))
    else:
        print('hiiii')
        cv2.fillPoly(road_poly, np.int_(road_poly_pts), (0, 0, 255))

    img = np.zeros_like(frame)
    # Colors in the left and right lane regions
    img[lefty, leftx] = [255, 0, 0]
    img[righty, rightx] = [0, 0, 255]
    img = cv2.addWeighted(img, 1, road_poly, 1, 0)
    unwarped = cv2.warpPerspective(img, Minv, (img.shape[1],img.shape[0]), flags=cv2.INTER_LINEAR)
    # crup unwarped
    mask = np.zeros_like(frame)
    mask[450::,:,:] = 255
    masked_unwarped = cv2.bitwise_and(unwarped, mask)

    out_image = cv2.addWeighted(frame, 1, masked_unwarped, 0.3, 0)

    cv2.putText(out_image, "Right curvature:        {:.2f}m".format(right_line.radius_of_curvature), (800,100), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,200),2)
    cv2.putText(out_image, "Distance from right:    {:.2f}m".format(right_line.line_base_pos), (800,150), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,200),2)

    cv2.putText(out_image, "Left curvature:        {:.2f}m".format(left_line.radius_of_curvature), (100,100), cv2.FONT_HERSHEY_PLAIN, 1.5, (250,0,0),2)
    cv2.putText(out_image, "Distance from left:    {:.2f}m".format(left_line.line_base_pos), (100,150), cv2.FONT_HERSHEY_PLAIN, 1.5, (250,0,0),2)
    cv2.putText(out_image, "Center offset:    {:.2f}m ".format(left_line.line_base_pos-1.85), (500,350), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,0,255),2)

    return out_image

def pipeline(frame, left_line, right_line):

    binary_img = binary_image(frame, mode="hsv_filter") #, mode="hsv_filter"
    binary_warped = bird_eye(binary_img, camera_parameters)
    leftx, lefty, rightx, righty = find_path(binary_warped, left_line.current_fit, right_line.current_fit)
    left_poly, right_poly = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

    left_line.get_newline(left_poly, leftx, lefty)
    right_line.get_newline(right_poly, rightx, righty)

    out_img = path_visualization(frame, camera_parameters["Minv"], left_line, right_line)
    return out_img, left_line, right_line

class Line():
    def __init__(self, y_size, x_size):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficientsof the last n fits of the line
        self.recent_fitted_poly = []
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
        self.ploty = np.linspace(0, y_size-1, y_size)
        self.center_point = np.int(x_size/2)
    def get_newline(self, new_fit, allx, ally):
        if len(new_fit) == 0:
            self.detected = False
            self.current_fit = [np.array([False])]
            self.diffs = np.array([0,0,0], dtype='float')
            self.allx = None
            self.ally = None
            self.recent_fitted_poly = self.recent_fitted_poly[1::]
            self.best_fit = np.average(np.array(self.recent_fitted_poly), axis=0)

            new_fitx = (self.best_fit[0]*self.ploty**2 + self.best_fit[1]*self.ploty + self.best_fit[2])

            self.recent_xfitted = self.recent_xfitted[1::]
            self.bestx = np.average(np.array(self.recent_xfitted), axis=0)

            fitted_m = np.polyfit(self.ploty * 80/720, self.bestx * 3.7/810, 2)
            y_eval = np.max(self.ploty * 80/720)
            self.radius_of_curvature = ((1 + (2*fitted_m[0]*y_eval + fitted_m[1])**2)**1.5)/(2*fitted_m[0])
            self.line_base_pos = np.absolute(self.center_point - self.bestx[-1]) * 3.7/810
        else:
            self.detected = True
            self.diffs = self.current_fit - new_fit
            self.current_fit = new_fit
            self.allx = allx
            self.ally = ally
            if len(self.recent_fitted_poly) == 10:
                self.recent_fitted_poly = self.recent_fitted_poly[1::]

            self.recent_fitted_poly.append(new_fit)
            self.best_fit = np.average(np.array(self.recent_fitted_poly), axis=0)

            new_fitx = (self.best_fit[0]*self.ploty**2 + self.best_fit[1]*self.ploty + self.best_fit[2])

            if len(self.recent_xfitted) == 10:
                self.recent_xfitted = self.recent_xfitted[1::]
            self.recent_xfitted.append(new_fitx)
            self.bestx = np.average(np.array(self.recent_xfitted), axis=0)

            fitted_m = np.polyfit(self.ploty * 80/720, self.bestx * 3.7/810, 2)
            y_eval = np.max(self.ploty * 80/720)
            self.radius_of_curvature = ((1 + (2*fitted_m[0]*y_eval + fitted_m[1])**2)**1.5)/(2*fitted_m[0])
            self.line_base_pos = np.absolute(self.center_point - self.bestx[-1]) * 3.7/810
        return self

# create left and right lines objects
left_line = Line(720,1280)
right_line = Line(720,1280)
# capture video file
cap = cv2.VideoCapture('project_video.mp4')
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output-project.avi',fourcc, 20.0, (1280,720))

start_time = time.time()
while True:
    _, frame = cap.read()
    if frame is None:
        break
    out_img, left_poly, right_poly = pipeline(frame, left_line, right_line)
    cv2.imshow('warped', out_img)
    out.write(out_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("--- %s seconds ---" % (time.time() - start_time))
#release video objects
cap.release()
out.release()
cv2.destroyAllWindows()
