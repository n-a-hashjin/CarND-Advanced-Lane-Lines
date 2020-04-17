## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)
[image0]: ./output_images/corners.JPG "camera calibration"
[image1]: ./output_images/undistort.JPG "Undistorted"
[image2]: ./output_images/warped_img.JPG "Birds-eye view"
[image3]: ./output_images/sobel_x.JPG "X-direction Sobel"
[image4]: ./output_images/color_filter.JPG "color filtered in HSV color space"
[image5]: ./output_images/binarized_img.JPG "comparing 3 methodes"
[image6]: ./output_images/histogram.JPG "Histogram"
[image7]: ./output_images/lane_pixel.JPG "detect left and right lane pixels"
[image8]: ./output_images/road_visualization.JPG "fit polynomial and plot toad path"
[image9]: ./output_images/radius_of_curvature.JPG "Find radius of curvature near vehicle"
[image10]: ./output_images/offset.JPG "distance from center"
[image11]: ./output_images/search_around_poly.JPG "search around poly to find lane lines pixels"
[image12]: ./output_images/unwarped_img.JPG "warp back on the road"
[image13]: ./output_images/out_put.JPG "appearance of output frames"
[image14]: ./output_images/undistorted_road.JPG "orginal and undistorted image from car camera"
[image15]: ./output_images/math_radius.JPG "raduis of curvature formula"

[video1]: ./output_videos/output_video.mp4 "Video"

---

### Camera Calibration

The code for this step is contained in the Second code cell of the IPython notebook after We import some useful and necessary packages like OpenCV, located in "./advance_lane_lines.ipynb" .  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.
It is also should be considered that our chessbord printout for calibration is consist of 9 by 6 corners, in other word our chessboard size is 9x6.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image0]
![alt text][image1]

### Pipeline (single images)

#### 1. Distortion-corrected image.

After in first step we get the camera paramiters now we can apply it to images from the camera and get undistorted images. For this purpose we use `cv2.undistort()`. Applying this method to a test image will result in this:

![alt text][image14]


#### 2. HSV color selection and sobel-x gradient to create a thresholded binary image.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at 5, 7 and 9th code cells of IPython notebook located in "./advance_lane_lines.ipynb").

In 5th code cell of IPython notebook we have applied a color filter in hsv color space `hsv_filter()`. By setting color range for upper and lower color spectrum of white and yellow colors we will be able to detect lane lines very Well. However in dark places or bad lighting conditions sometimes it fails to detect lane lines.
Here's an example of my output for this function.

![alt text][image4]

In 7th code cell of IPython notebook there is a sobelx function, `name sobel_x()`. Sobelx direction derivation is performed on a grayscale image of undistorted image. In IPython notebook it is performed and demonstrated for all images in "./test_images" directory. Sobelx will find vertical lines very well, however under shadowing or different colors and changes in road it shows unwanted vertical lines.
Here's an example of my output for this function.

![alt text][image3]

In 9th code cell we have combined these 2 methods and the results of these 3 binarized image is shown in below example:

![alt text][image5]

#### 3. Perspective transform

The code for my perspective transform includes a function called `warp_image()`, which appears in lines 1 through 4 in the 22nd code cell of IPython notebook `advance_lane_lines.ipynb` (./advance_lane_lines.ipynb).  The `warp_image()` function takes as inputs an image (`img`), and returns warped image or birds-eye view.
To use `warp_image()` functio we need first calculate Transform Matrices. I chose the hardcode the source and destination points in the following manner:

```python
# src and dst for perspective transform
src = np.float32([[583,455],[219,690],[1059,690],[695,455]])
dst = np.float32([[280,0],[280,720],[1000,720],[1000,0]])

M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)
```
**the selected points are base on an isosceles trapezoid to represent strait lines in parallel form and vertical shape in warped image**

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image2]

#### 4. Lane-line pixels identification and fit theme a polynomial

Then I used histogram fiture on bottom half of warped image to determine where is left and right lane-lines located. It will give a start point to search the image for rest of points. We devide image to 9 horizontal slice and in each half of them searching for lane line pixel, if ther was more than a threshold amount pixel detected then we relocalize the window center. Continuing this returns left and right lane lines. In below you see histogram and then detected pixels. The green boxes are the search window described before.

![alt text][image6]

![alt text][image7]

However this process can be slow, so in next coming frame that because of slightly small time gap between them we can consider it almost very close to current frame. Due to this assumption for next frame we will search around last fited polynomial. In below you see the fitted poly in yellow and the area of search for next frame in transparent green.

![alt text][image11]

#### 5. Radius of curvature of the lane and center position offset

Radius of curvature for a curve of second order polynomial is being calculated from below formula.

![alt text][image15]

I did this in 17th code cell of IPython notebook `advance_lane_lines.ipynb`

```python
xm_per_pixel = 3.7/700
ym_per_pixel = 30/720
ym = ploty * ym_per_pixel

left_xm =  left_fitx * xm_per_pixel
left_fitxm = np.polyfit(ym, left_xm, 2)

right_xm =  right_fitx * xm_per_pixel
right_fitxm = np.polyfit(ym, right_xm, 2)

y_eval = np.max(ym)
left_line_radius_of_curvature = ((1 + (2*left_fitxm[0]*y_eval + left_fitxm[1])**2)**1.5)/(2*left_fitxm[0])
right_line_radius_of_curvature = ((1 + (2*right_fitxm[0]*y_eval + right_fitxm[1])**2)**1.5)/(2*right_fitxm[0])
```
The visualizing results in:

![alt text][image9]

In 18th code cell of the same file we calculate the center offset. We assume that the center of image is the center of car and then we need to find the center of left and right side lanes and do a subtraction on these two number of pixel.

```python
y_bottom = out_img.shape[0]
leftx_bottom = left_fit[0]*y_bottom**2 + left_fit[1]*y_bottom +left_fit[2]
rightx_bottom = right_fit[0]*y_bottom**2 + right_fit[1]*y_bottom +right_fit[2]
center_of_lane_lines = (leftx_bottom + rightx_bottom)//2
center_of_car = out_img.shape[1]//2
offset = (center_of_car - center_of_lane_lines)* xm_per_pixel
```

![alt text][image10]

#### 6. Plotted back down onto the road!

I implemented this step in 19th code cell `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

```python
detected_unwarped = cv2.warpPerspective(detected_warped, Minv, (img.shape[1],img.shape[0]), flags=cv2.INTER_LINEAR)
result_image = cv2.addWeighted(img[:,:,::-1], 1, detected_unwarped, 1, 0)
plt.imshow(result_image)
```

We had already ```Minv``` from this line:

```python
Minv = cv2.getPerspectiveTransform(dst, src)
```
The first and second below images show the warped image and unwarped image.

![alt text][image8]

![alt text][image12]

---
#### 7. Keep Track of Everything

To keep track of information in every iteration we have defined ```Line()``` class. it saves most important information in every iteration and keep some of them for next 10 iteration. It automatically run an average function on last polinomial of line for example. In this way we can avoid jitter or loosing track of line when in one or a few frames the system fails to detect line. This class is defined in 21st code cell of IPython notebook.


### Pipeline (video)

#### 1. Applying the Pipeline on Video

each frame looks like![alt text][image13]
Here's a [link to my video result](./output_videos/output_video.mp4)

---

### Discussion

I have applied above steps and after running the pipeline under different binarization threshold and methods I founded out that it can be very critical criteria. When the lighting conditions changes or the lane lines fade or any change in color of road can affect my pipline effectiveness. Applying another filter for reject unrelated and unwanted noises like the given examples can be helpfull. This algorithm is also very likely to fail for fast changes in direction of lane lines. It could need more smarter decision making maybe!
