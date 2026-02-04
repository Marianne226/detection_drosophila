import numpy as np
import cv2 as cv
from dataAcq_func_2flies import *


#in_path = '/Users/mariannecivitardevol/Documents/EPFL/Masters/projet semestre1/tracking_code/Training classifier/Get_Data/20240129-154411.mp4'
in_path = '/Users/mariannecivitardevol/Documents/EPFL/Masters/projet semestre1/tracking_code/Training classifier/new_get_data/20240318-165751.mp4'


# Output path, where the images are saved
folder_path = '/Users/mariannecivitardevol/Documents/EPFL/Masters/projet semestre1/Finalised_code/classifiers_creation/get_data_functions/new_data/'
#FOR TEST
#folder_path = '/Users/mariannecivitardevol/Documents/EPFL/Masters/projet semestre1/tracking_code/Training classifier/test_data/'

#for video reading
cap = cv.VideoCapture(in_path)
if cap.isOpened() == False:
    raise Exception('Video file cannot be read! Please check in_path to ensure it is correctly pointing to the video file')
raw_imgs=[]


# 2 flies max
t_id = ['A', 'B']

number = 0
count = 0
while(True):
    # Capture frame-by-frame
    if count%10100 == 0: #frequency at which we want to collect a frame
        # Set the frame position
        cap.set(cv.CAP_PROP_POS_FRAMES, count)
        ret, frame = cap.read()
        if ret == True:
            if count == 0:
                raw_imgs_gray = cv.cvtColor(frame.copy(), cv.COLOR_BGR2GRAY)

                n_inds = 2

                print('Preprocessing: Finding Number of Flies using frame 0')
                _, meanArea = preprocessing(frame)

                meas_last = list(np.zeros((n_inds,2)))
                meas_now = list(np.zeros((n_inds,2)))
                
            print('Processing frame: ' + str(count), end='\r')
 
            frame = hide_walls(frame.copy())

            # Convert Frame to Grayscale
            gray = cv.cvtColor(frame.copy(), cv.COLOR_BGR2GRAY)

            # Thresholding Image
            thresh = thresholdImage(gray.copy())
            
            # Detecting Flies
            meas_save = meas_now.copy()
            final, contours, meas_last, meas_now = detect_and_draw_contours_new(frame, thresh, meanArea, meas_now)   
            if len(contours) < n_inds:
                contours_new, meanArea_New, correct = correctDetection(thresh, n_inds)
                if correct:
                    final = cv.drawContours(frame.copy(), contours_new, -1, (0,255,0), 3)
                    meas_now = reformatted(meas_last, meas_now)
                    contours = contours_new
                print('meas_save of frame {} after correction is {}'.format(count, meas_save))
                print('meas_last of frame {} after correction is {}'.format(count, meas_last))
                print('meas_now of frame {} after correction is {}'.format(count, meas_now))
            
            # If we cannot detect any flies, raise exception
            if len(contours)==0:
                print('Breaking at frame no. {}'.format(count))
                print('Because contour length is 0')
                print('Mean area at this point is {}'.format(meanArea_New))
                raise Exception('No contours found')     

            # If we could not find the correct number of flies, break out of loop instead of calling K-Means
            if len(meas_now) != n_inds and len(contours)>0:
                meas_now = meas_last
                print('Breaking out of loop before K-Means')
                continue

            #Approximate contour to ellipse and extract image
            for contour in contours:
                final = approximate_to_ellipse(contour,frame)
                filename = f'fly{number}.jpg'
                file_path = folder_path + filename
                cv.imwrite(file_path, final)
                number += 1
        else:
            break

    count += 1
