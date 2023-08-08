import cv2
import os
import numpy as np

## load in images ## 

path = 'C:/Users/tflom/OneDrive/Desktop/matrixdegradation/T2_rawimages'
files = os.listdir(path)
for file in files: 
    file_path = os.path.join(path, file) 
    file_name = os.path.basename(file_path)
    
    ## convert and sharpen image ##

    original_img = cv2.imread(file_path)       
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    brightness = 3
    contrast = 1.4
    sharpened_img = cv2.addWeighted(original_img, contrast, np.zeros(original_img.shape, original_img.dtype), 0, brightness)

    ## gray threshold with high fidelity ##

    hardcode_gray_thresh = 223 # adjust thresholding as needed
    gray_img = cv2.cvtColor(sharpened_img, cv2.COLOR_RGB2GRAY)
    (thresh, bw_img) = cv2.threshold(gray_img, hardcode_gray_thresh, 255, cv2.THRESH_BINARY)

    ## invert grayscale and colors & color all non-black pixels yellow ##

    intermed_img = cv2.cvtColor(bw_img, cv2.COLOR_GRAY2RGB)
    hsv_img = cv2.cvtColor(intermed_img, cv2.COLOR_RGB2HSV)
    white_thresh = np.array([0, 0, 0])
    cvt_mask = cv2.inRange(hsv_img, white_thresh, white_thresh)
    intermed_img[cvt_mask > 0] = (255, 0, 0)
    inverted_mask = cv2.bitwise_not(intermed_img)

    ## convert to hsv and specify aggregate dimension treshold ##

    hsv = cv2.cvtColor(inverted_mask, cv2.COLOR_BGR2HSV)
    min_h = 32
    min_w = 32

    ## define yellow bounds ##

    lower_yellow = np.array([20, 100, 100]) 
    upper_yellow = np.array([30, 255, 255])

    ## generate mask for yellow objects, find contours ##

    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ## loop through the contours and find regions larger than the threshold size ## 

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= min_h and h >= min_w:
            # Change color of yellow pixels to blue
            blue_pixels = (yellow_mask[y:y+h, x:x+w] == 255)
            inverted_mask[y:y+h, x:x+w][blue_pixels] = [255, 0, 0] 

    ## save masks ## 
    
    cv2.imwrite('final_mask_' + file, inverted_mask)




cv2.waitKey(0)  
cv2.destroyAllWindows()