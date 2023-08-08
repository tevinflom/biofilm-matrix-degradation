import cv2
import os
import numpy as np
import csv

names = []
percents = []

## load in the images ## 

path = 'C:/Users/tflom/OneDrive/Desktop/dispersiondetection/computer-vision-matrix-degradation/T1_preprocessed'
files = os.listdir(path)
for file in files: 
    file_path = os.path.join(path, file)
    file_name = os.path.basename(file_path)
    names.append(file_name)

    ## specify image path and convert to HSV ## 

    img = cv2.imread(file_path)
    HSVimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ## threshold yellow and blue pixels ##

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([130, 255, 255])

    ## create masks for yellow and blue regions ##

    yellow_mask = cv2.inRange(HSVimg, lower_yellow, upper_yellow)
    blue_mask = cv2.inRange(HSVimg, lower_blue, upper_blue)

    ## count the number of yellow & blue pixels in the mask ##

    num_yellow_pixels = np.count_nonzero(yellow_mask)
    num_blue_pixels = np.count_nonzero(blue_mask)


    ## translate pixel ratio to % biomass as aggregates ##

    total_annotpixels = num_yellow_pixels + num_blue_pixels
    aggregated_ratio = num_blue_pixels / total_annotpixels
    aggregated_percentage = aggregated_ratio * 100 
    percents.append(aggregated_percentage)
    

    # img_2_percentage = {(file_name) : (aggregated_percentage)}
    # img_2_percentage[file_name].append(aggregated_percentage)


    #print(f'{file_name} : {aggregated_percentage}')
#print(f'{names}, {percents}')

 




    ## save as csv ##
# img_2_percentage = {(file_name) : (aggregated_percentage)}
# img_2_percentage = {}
# img_2_percentage[file_name] = aggregated_percentage

img_2_percentage = {names[i]:percents[i] for i in range(len(names))}
for file in files:

    img_2_percentage.update({file_name:aggregated_percentage})
    with open('output.csv', 'w') as output:
        writer = csv.writer(output)
        for file_name, aggregated_percentage in img_2_percentage.items():
            writer.writerow([file_name, aggregated_percentage])