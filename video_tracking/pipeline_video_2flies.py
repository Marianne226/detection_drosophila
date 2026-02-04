import pandas as pd
import numpy as np
import cv2 as cv
import random
import os
import time
import math
from torchvision import transforms
from PIL import Image
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

class Experiment:
    def __init__(self,exp_dir):

        self.in_path = exp_dir
        self.fps = 0
        self.walls = None
        self.trajectories = None
        self.meas_now = list(np.zeros((2,2)))
        self.meas_last = list(np.zeros((2,2)))
        self.df = pd.DataFrame()
        self.kMeansFrames = []
        self.fly_id = 0
        self.avg_intensity_previous = []
        self.previous_predicted = []
        self.corrected_times = 0
        self.corrected_times_pred = 0
        self.predicted_way = 1
        self.robot_size = (10,10)
        self.robotID = 1
        self.reset = False
        self.space_radius = 0
        self.metrics = []

    def reformatted(self, last, now):
        lstIDS = []
        for i in range(len(last)):
            distList = [np.linalg.norm(np.array(last[i])-np.array(now[j])) for j in range(len(now))]
            lstIDS.append(distList.index(min(distList)))
        now_reformatted = [now[i] for i in lstIDS]
        return now_reformatted
    
    def correctDetection(self, frame, num):
        """
        Corrects the detection in the given frame by separating shapes using erosion
        and dilation, and checking if the number of contours matches the expected number.

        Parameters:
        - frame (numpy.ndarray): The input image frame.
        - num (int): The expected number of contours.

        Returns:
        - dilated_contours (list): List of dilated contours found in the frame.
        - meanArea (float): The mean area of the contours.
        - corrected (bool): True if the number of contours matches the expected number, False otherwise.
        """

        corrected = False

        # Erode the frame to separate the shapes
        kernel1 = np.ones((50, 50), np.uint8)
        eroded = cv.erode(frame, kernel1, iterations=1)

        # Find contours of the eroded shapes
        contours, _ = cv.findContours(eroded, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        dilated_contours = []

        for contour in contours:
            # Create a mask for each contour
            mask = np.zeros_like(frame)
            cv.drawContours(mask, [contour], 0, 255, -1)  # Draw the contour as a filled white shape

            # Dilate the mask
            dilated_mask = cv.dilate(mask, np.ones((50, 50), np.uint8), iterations=1)
            dilated_contours_list, _ = cv.findContours(dilated_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            
            # Add dilated contours to the list
            for contour2 in dilated_contours_list:
                dilated_contours.append(contour2)

        # Calculate the area of each contour
        areas = [cv.contourArea(contour) for contour in contours]
        meanArea = np.mean(np.array(areas))

        # Keep contours with area greater than 80% of the mean area
        contoursToKeep = [contour for contour in contours if cv.contourArea(contour) > 0.8 * meanArea]

        # Check if the number of contours matches the expected number
        if len(contoursToKeep) == num:
            corrected = True

        return dilated_contours, meanArea, corrected
    
    def thresholdImage(self, frame):
        '''
        Performs binary thresholding of the image

        Parameters
        ----------
        - frame: ndarray, shape(n_rows, n_cols)
            grayscale frame with background subtracted

        Returns
        -------
        - thresh: ndarray, shape(n_rows, n_cols)
            binary thresholded image of the flies such that the flies are white and the rest of the image 
            is black
        '''
        # Blur to Clean Noise 
        noiseCleaned = cv.blur(frame,(25,25))

        # Find Threshold
        ret, thresh = cv.threshold(noiseCleaned,50,255,cv.THRESH_BINARY)

        return thresh
    
    def get_fly_orientation(self, frame_output, frame_analysis, contour):
        '''
        Finds the orientation of the flies and outputs the angle with respect to the vertical axis [0,180] and [0,-180].
        The function checks if the found position makes sense and prevents error glitches
        It also outputs masks and frames that are useful for another function.

        Parameters
        ----------
        - frame_output : ndarray, shape (n_rows, n_cols, 3)
            BGR frame for adding shapes onto it (indicating visually where the fly's head is).
        - frame_analysis : ndarray, shape (n_rows, n_cols)
            Grayscale frame used for analysis.
        - contour : ndarray
            Contour of the fly.

        Returns
        -------
        - fly_angle : float
            The angle of the fly with respect to the vertical axis: [0,180] and [0,-180].
        - translated_rotated_mask1 : ndarray, shape (n_rows, n_cols)
            Translated, rotated, and cut (top part) mask of the approximation to an ellipse of the fly's contour.
        - translated_rotated_mask2 : ndarray, shape (n_rows, n_cols)
            Translated, rotated, and cut (bottom part) mask of the approximation to an ellipse of the fly's contour.
        - rotated_mask : ndarray, shape (n_rows, n_cols)
            Translated and rotated mask of the fly's approximation to an ellipse without being cut.
        - rotated_frame : ndarray, shape (n_rows, n_cols)
            Translated and rotated frame of the fly's approximation to an ellipse without being cut.
        - center : tuple
            Position of the center of the fly (approximated to an ellipse).
        - ma : float
            Full length of the long size of the approximated ellipse.
        '''

        # Approximate contour to oval/ellipse
        ellipse = cv.fitEllipse(contour)

        # Create an empty mask
        mask = np.zeros_like(frame_analysis)

        # Draw the ellipse on the mask
        cv.ellipse(mask, ellipse, (255, 255, 255), -1)

        # Extract the parameters of the ellipse
        (x, y), (MA, ma), angle_init = ellipse  # coordinates of center (x,y) and major/minor axes length
        center = (int(x), int(y))

        if angle_init > 90:
            angle_rot = angle_init - 180
            angle = 180 - angle_init
            fly_angle = -angle  # this is the angle of the fly with respect to the vertical axis (clockwise)
            angle_pos = (360 + fly_angle) % 180
        else:
            angle_rot = angle_init
            angle = angle_init
            fly_angle = angle  # this is the angle of the fly with respect to the vertical axis (clockwise)
            angle_pos = fly_angle % 180

        # Create 2 masks and fit them to the ellipse approximation
        half1_mask = np.zeros_like(frame_analysis)
        half2_mask = np.zeros_like(frame_analysis)
        cv.ellipse(half1_mask, ellipse, (255, 255, 255), -1)
        cv.ellipse(half2_mask, ellipse, (255, 255, 255), -1)

        # Translate the mask to the center of the frame
        translation_matrix = np.float32([[1, 0, (frame_analysis.shape[1] / 2 - x)], [0, 1, (frame_analysis.shape[0] / 2 - y)]])
        translated_frame = cv.warpAffine(frame_analysis, translation_matrix, (frame_analysis.shape[1], frame_analysis.shape[0]))
        translated_mask1 = cv.warpAffine(half1_mask, translation_matrix, (frame_analysis.shape[1], frame_analysis.shape[0]))
        translated_mask2 = cv.warpAffine(half2_mask, translation_matrix, (frame_analysis.shape[1], frame_analysis.shape[0]))

        # Rotate the masks and frame
        rotation_matrix = cv.getRotationMatrix2D((frame_analysis.shape[1] / 2, frame_analysis.shape[0] / 2), angle_rot, 1.0)
        translated_rotated_mask1 = cv.warpAffine(translated_mask1, rotation_matrix, (frame_analysis.shape[1], frame_analysis.shape[0]))
        rotated_mask = translated_rotated_mask1.copy()
        translated_rotated_mask2 = cv.warpAffine(translated_mask2, rotation_matrix, (frame_analysis.shape[1], frame_analysis.shape[0]))
        rotated_frame = cv.warpAffine(translated_frame, rotation_matrix, (frame_analysis.shape[1], frame_analysis.shape[0]))

        # Cut the masks horizontally
        translated_rotated_mask1[int(frame_analysis.shape[0] / 2 - (ma / 4)):, :] = 0
        translated_rotated_mask2[:int(frame_analysis.shape[0] / 2 + (ma / 4)), :] = 0

        # Rotate back to original
        rotation_matrix2 = cv.getRotationMatrix2D((frame_analysis.shape[1] / 2, frame_analysis.shape[0] / 2), -angle_rot, 1.0)
        translated_rotated_back1 = cv.warpAffine(translated_rotated_mask1, rotation_matrix2, (frame_analysis.shape[1], frame_analysis.shape[0]))
        translated_rotated_back2 = cv.warpAffine(translated_rotated_mask2, rotation_matrix2, (frame_analysis.shape[1], frame_analysis.shape[0]))

        # Translate back to original
        translation_matrix2 = np.float32([[1, 0, (x - frame_analysis.shape[1] / 2)], [0, 1, (y - frame_analysis.shape[0] / 2)]])
        rotated_back1 = cv.warpAffine(translated_rotated_back1, translation_matrix2, (frame_analysis.shape[1], frame_analysis.shape[0]))
        rotated_back2 = cv.warpAffine(translated_rotated_back2, translation_matrix2, (frame_analysis.shape[1], frame_analysis.shape[0]))

        # Apply mask to image
        masked_image1 = cv.bitwise_and(frame_analysis, rotated_back1)
        masked_image2 = cv.bitwise_and(frame_analysis, rotated_back2)

        # Calculate the average pixel intensity for each half
        average_intensity_half1 = np.mean(masked_image1)
        average_intensity_half2 = np.mean(masked_image2)

        if average_intensity_half1 > average_intensity_half2:
            avg_intensity_half = 1
        else:
            avg_intensity_half = 2

        # Check if the head position makes sense
        if len(self.avg_intensity_previous) >= 5 and (angle_pos < 70 or angle_pos > 110):
            possible, self.corrected_times = double_check(self.avg_intensity_previous, avg_intensity_half, self.corrected_times, 15)
            if not possible:
                avg_intensity_half = avg_intensity_half % 2 + 1  # convert a 2 to a 1 and vice versa

        # Add head position to array for checking
        self.avg_intensity_previous = add_to_array(self.avg_intensity_previous, avg_intensity_half, 10)

        # Determine where the head is
        if avg_intensity_half == 1:
            if angle_init > 90:
                pointx = int(-abs((ma / 2) * math.cos(math.pi / 2 - angle * math.pi / 180)) + x)
                pointy = int(-abs((ma / 2) * math.sin(math.pi / 2 - angle * math.pi / 180)) + y)
            else:
                pointx = int(abs((ma / 2) * math.cos(math.pi / 2 - angle * math.pi / 180)) + x)
                pointy = int(-abs((ma / 2) * math.sin(math.pi / 2 - angle * math.pi / 180)) + y)
            cv.circle(frame_output, (pointx, pointy), 5, (255, 0, 0), -1, cv.LINE_AA)
            cv.circle(rotated_back1, (pointx, pointy), 5, (255, 0, 0), -1, cv.LINE_AA)
        else:
            if angle_init > 90:
                pointx = int(abs((ma / 2) * math.cos(math.pi / 2 - angle * math.pi / 180)) + x)
                pointy = int(abs((ma / 2) * math.sin(math.pi / 2 - angle * math.pi / 180)) + y)
                fly_angle = 360 + fly_angle - 180
            else:
                pointx = int(-abs((ma / 2) * math.cos(math.pi / 2 - angle * math.pi / 180)) + x)
                pointy = int(abs((ma / 2) * math.sin(math.pi / 2 - angle * math.pi / 180)) + y)
                fly_angle = fly_angle + 180 - 360
            cv.circle(frame_output, (pointx, pointy), 5, (255, 0, 0), -1, cv.LINE_AA)
            cv.circle(rotated_back2, (pointx, pointy), 5, (255, 0, 0), -1, cv.LINE_AA)

        return fly_angle, translated_rotated_mask1, translated_rotated_mask2, rotated_mask, rotated_frame, center, ma
    
    def find_vertical_horizontal_walls(self, meas_now):
        '''
        Separates the vertical from the horizontal walls in the walls list.

        Parameters
        ----------
        - meas_now : tuple, shape (1, 2)
            The center position of the fly.

        Returns
        -------
        - vertical_walls : list of ndarray
            Array containing the end points of the vertical walls.
        - horizontal_walls : list of ndarray
            Array containing the end points of the horizontal walls.
        '''

        # Initialize arrays to store coefficients of line equations
        a = np.zeros(len(self.walls))
        b = np.zeros(len(self.walls))
        c = np.zeros(len(self.walls))

        # Determine each wall's line equation
        for i in range(len(self.walls)):
            a0 = self.walls[i][0][0]
            b0 = self.walls[i][0][1]
            if i != len(self.walls) - 1:
                a1 = self.walls[i + 1][0][0]
                b1 = self.walls[i + 1][0][1]
            else:
                a1 = self.walls[0][0][0]
                b1 = self.walls[0][0][1]
            if (a1 - a0) != 0:
                a[i] = 0
                b[i] = -1
                c[i] = b0
            else:
                a[i] = -1
                b[i] = 0
                c[i] = a0

        distance_to_vert = []
        vertical_walls = []
        horizontal_walls = []
        distance_to_horiz = []

        # Determine whether it is vertical or horizontal wall
        for i in range(len(a)):
            # Vertical
            if b[i] == 0:
                dist = abs(a[i] * meas_now[0] + b[i] * meas_now[1] + c[i]) / math.sqrt(a[i] ** 2 + b[i] ** 2)
                vertical_walls.append(self.walls[i])
                distance_to_vert.append(dist)
            # Horizontal
            else:
                dist = abs(a[i] * meas_now[0] + b[i] * meas_now[1] + c[i]) / math.sqrt(a[i] ** 2 + b[i] ** 2)
                horizontal_walls.append(self.walls[i])
                distance_to_horiz.append(dist)

        return vertical_walls, horizontal_walls

    def process_video(self, pca_reloaded, modelUp_or_Down, model_CNN, cnn, svm, both):

        # ----------------------------------------------------

        # Reading the video, Name of source video and paths
        out_path = self.in_path.replace('.mp4', '_tracked.mp4')

        cap = cv.VideoCapture(self.in_path)
        if cap.isOpened() == False:
            raise Exception('Video file cannot be read! Please check in_path to ensure it is correctly pointing to the video file')
        width  = int(cap.get(3))
        height = int(cap.get(4))
        fourcc = cv.VideoWriter_fourcc(*"mp4v") 
        self.fps = int(cap.get(5))


        output_framesize = (width, height)
        out = cv.VideoWriter(filename = out_path, fourcc = fourcc, fps=self.fps, frameSize = output_framesize, isColor = True)

        # ----------------------------------------------------
        #define the number of elements --> flys and robot
        n_inds = 2

        #initialise the needed variables
        raw_imgs=[]
        start_ang_h = 0 
        finish_ang_h = 0
        fly_angle = np.zeros(n_inds)
        fly_facing = np.zeros(n_inds)
        # ----------------------------------------------------

        # 2 flies max
        t_id = ['A', 'B']

        #Colors of the ids
        colors = [(0,0,255),(0,255,255)]

        

        # ----------------------------------------------------
        c = 0
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                #if video processing
                raw_imgs.append(frame)
            else:
                break
            c+=1


        print('Ready to Process frames')
        for frame_number, frame in enumerate(raw_imgs):
            print('processing frame ', frame_number)
            frame_color_output = frame.copy()
            frame_gray_analysis = cv.cvtColor(frame.copy(), cv.COLOR_BGR2GRAY)

            if frame_number == 2:
                # Find the walls in the frames
                self.walls = get_wallsNew(frame_gray_analysis)

            if frame_number >= 2:
                # Hide the walls
                frame_no_walls = hide_walls(frame_gray_analysis)

                # Thresholding Image
                frame_no_walls_thresh = self.thresholdImage(frame_no_walls)

                # Detecting Flies
                meas_save = self.meas_now.copy()
                contours, self.meas_last, self.meas_now = detect_contours(frame_no_walls_thresh, self.meas_now, frame_no_walls, frame_gray_analysis, frame)          
                
                # Check if the found contours are correct
                if len(contours) < n_inds:
                    print('trying to correct detection')
                    contours_new, meanArea_New, correct = self.correctDetection(frame_no_walls_thresh, n_inds)
                    if correct:
                        cv.drawContours(frame_color_output, contours_new, -1, (0,255,0), 3)
                        contours = contours_new
                        self.meas_now = self.reformatted(self.meas_last, self.meas_now)
                    print('meas_save of frame {} after correction is {}'.format(frame_number, meas_save))
                    print('meas_last of frame {} after correction is {}'.format(frame_number, self.meas_last))
                    print('meas_now of frame {} after correction is {}'.format(frame_number, self.meas_now))
                else:
                    cv.drawContours(frame_color_output, contours, -1, (0,255,0), 3)

                # If we cannot detect any flies, raise exception
                if len(contours) == 0:
                    print('Breaking at frame no. {}'.format(frame_number))
                    print('Because contour length is 0')
                    print('Mean area at this point is {}'.format(meanArea_New))
                    raise Exception('No contours found')   

                right_number_contours = True

                # If we could not find the correct number of flies, break out of loop instead of calling K-Means
                if len(self.meas_now) != n_inds and len(contours) > 0:
                    self.kMeansFrames.append(frame_number)
                    self.meas_now = self.meas_last
                    right_number_contours = False
                    print('Breaking out of loop before K-Means')

                # If we found the correct number of contours, we can do hungarian algorithm
                if len(self.meas_now) == n_inds and len(contours) >= 2:
                    reorder_time = time.time()
                    # Hungarian Algorithms for ID Assignment
                    row_ind, col_ind = hungarian_algorithm(self.meas_last, self.meas_now)
                    # Draw centroid of the contours
                    contours, self.meas_now = reorder_and_draw(frame_color_output, colors, n_inds, col_ind, contours, frame_number, self.meas_now)
                    if frame_number > 20 and len(contours[self.fly_id]) < 5:
                        right_number_contours = False
                    reorder_elapsed_time = time.time() - reorder_time

                if right_number_contours == True:
                    nb = 0
                    for contour in contours : 
                        # Find the fly's head position
                        fly_angle[nb], translated_rotated_mask1, translated_rotated_mask2, rotated_mask, rotated_frame, center, fly_size = self.get_fly_orientation(frame_color_output, frame_no_walls, contour)

                        # Find if the fly is facing up or down
                        if (frame_number % 10) == 0:
                            final_ML = approximate_to_ellipse(frame_no_walls, translated_rotated_mask1, translated_rotated_mask2, rotated_mask, rotated_frame)
                            fly_facing[nb] = predict_facing(cnn, svm, both, pca_reloaded, modelUp_or_Down, model_CNN, final_ML)

                            # Check if the position makes sense compared to the previous ones
                            '''if len(self.previous_predicted) >= 5:
                                possible, self.corrected_times_pred = double_check(self.previous_predicted, self.predicted_way, self.corrected_times_pred,5)
                                if not possible:
                                    self.predicted_way = self.predicted_way % 2 + 1

                            # Add facing position to array for further checking
                            self.previous_predicted = add_to_array(self.previous_predicted, self.predicted_way, 5)'''

                        if fly_facing[nb] == 1:
                            #we see the fly's stomach -> green
                            cv.circle(frame_color_output, center, 5, (0,255,0), -1, cv.LINE_AA)
                        else:
                            #We see the fly's back -> yellow
                            cv.circle(frame_color_output, center, 5, (0,255,255), -1, cv.LINE_AA)
                        nb+=1


                    # Create output dataframe
                    for i in range(n_inds):
                        new_data = [[frame_number, self.meas_now[i][0], self.meas_now[i][1], t_id[i], fly_angle[i], fly_facing[i]]]
                        new_rows = pd.DataFrame(new_data)
                        self.df = pd.concat([self.df, new_rows], ignore_index=True)

                    self.KMeansProcessedFrames = self.kMeansFrames
                    
            # Save the resulting frame
            out.write(frame_color_output)

        # When everything done, release the capture
        cap.release()
        out.release()

        df = pd.DataFrame(np.matrix(self.df), columns = ['frame', 'pos_x', 'pos_y', 'id', 'angle', 'facing'])

        # Dataframe manipulations to clean up format
        df['id'] = pd.Categorical(df.id)
        df['id'] = df.id.cat.codes
        df.frame = pd.to_numeric(df.frame)
        df.pos_x = pd.to_numeric(df.pos_x)
        df.pos_y = pd.to_numeric(df.pos_y)
        df.angle = pd.to_numeric(df.angle)
        df = df.pivot(index = 'frame', columns = 'id', values = ['pos_x', 'pos_y','angle'] )
        df = df.reorder_levels(['id', None], axis = 1)
        df = df.sort_index(axis = 1, level = 'id')

        self.df = df
        return 


def prepare_for_ML(gray_ellipse, pca_reloaded, cnn, svm):
    '''
    Prepares the frame for evaluation with the SVM or CNN classifier depending on the chosen input
    It performs histogram equalization for brightness issues, flattens the array, and performs PCA.

    Parameters
    ----------
    - gray_ellipse : ndarray, shape (n_rows, n_cols)
        Frame containing only the approximated ellipse of the fly.
    - pca_reloaded : PCA object
        Trained PCA algorithm loaded for fitting.
    - cnn : bool
        Flag indicating whether to prepare the frame for CNN evaluation.
    - svm : bool
        Flag indicating whether to prepare the frame for SVM evaluation.

    Returns
    -------
    - pca_image : ndarray, shape (1, n_features=2)
        Original frame to which histogram equalization has been applied, flattened, and PCA applied.
    '''

    if cnn:
        # If using CNN, apply preprocessing transformations
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Convert the grayscale ellipse to color
        color_ellipse = cv.cvtColor(gray_ellipse, cv.COLOR_GRAY2BGR)
        # Convert to PIL image
        image = Image.fromarray(color_ellipse)
        # Apply preprocessing transformations
        input_tensor = preprocess(image)
        # Prepare for CNN input format
        pca_image = input_tensor.unsqueeze(0)
    elif svm:
        
        # If using SVM, apply histogram equalization
        gray_histogrammed = cv.equalizeHist(gray_ellipse)

        # Reshape into a flat array
        height, width = gray_histogrammed.shape
        gray_flat = gray_ellipse.reshape(1, height * width)

        # Normalise the image
        normalized_image = (gray_flat - gray_flat.mean(axis=1, keepdims=True)) / gray_flat.std(axis=1, keepdims=True)

        # Perform PCA
        pca_image = pca_reloaded.transform(normalized_image)

    return pca_image


def hide_walls(frame):
    '''
    Hides the walls from the frame for thresholding and contour detection to avoid errors.

    Parameters
    ----------
    - frame : ndarray, shape (n_rows, n_cols)
        Grayscale frame used for analysis.

    Returns
    -------
    - no_walls : ndarray, shape (n_rows, n_cols)
        Frame to which we have made the borders dark in order to remove any traces of the walls.
'''
    # Create a blank image with the same dimensions as the frame
    blank = np.zeros_like(frame)

    #dimensions of the border
    border_width = 25

    # Create a white rectangle in the center of the blank image
    cv.rectangle(blank, (border_width, border_width),
                  (frame.shape[1] - border_width, frame.shape[0] - border_width), (255, 255, 255), -1)
    
    # Apply mask to frame
    no_walls = cv.bitwise_and(frame, blank)
    
    return no_walls


def get_wallsNew(frame_for_walls):
    '''
    Finds the position of the walls in the frame. If no walls are found, we assume that the frame's limits are the walls.

    Parameters
    ----------
    - frame_for_walls : ndarray, shape (n_rows, n_cols)
        Grayscale frame used for analysis.

    Returns
    -------
    - walls : ndarray, shape (4, 1, 2)
        Contours representing the walls of the frame, approximated to a square.
        Each contour consists of 4 points, defining the corners of the square.
    '''

    # Crop the image to a square if it isn't
    width, height = frame_for_walls.shape
    shift = False
    if width != height:
        shift = True
        if width > height:
            frame_for_walls = frame_for_walls[(width // 2) - (height // 2):(width // 2) + (height // 2), :]
            shift_x = 0
            shift_y = (width - height) // 2
        else:
            frame_for_walls = frame_for_walls[:, (height // 2) - (width // 2):(height // 2) + (width // 2)]
            shift_x = (height - width) // 2
            shift_y = 0

    # Find contours
    contoursWall, _ = cv.findContours(frame_for_walls, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours_sorted = sorted(contoursWall, key=cv.contourArea, reverse=True)

    # Approximate the contour to a square
    epsilon = 0.02 * cv.arcLength(contours_sorted[0], True)
    approximated = cv.approxPolyDP(contours_sorted[0], epsilon, closed=True)

    width2, height2 = frame_for_walls.shape
    if cv.contourArea(approximated) > width2 * height2 * 0.99:
        print('Found walls')
        walls = approximated
        if shift:
            walls[:, 0, 0] += shift_x
            walls[:, 0, 1] += shift_y

    return walls


def determine_fly_robot(meas_now):
    '''
    Ask the user to tell which one of the two elements in the frame corresponds to the fly 
    and then keeps its ID to separate them.

    Parameters
    ----------
    - meas_now : ndarray of tuples
        Array containing the center position of the fly and the robot.
        Each row corresponds to the position of one element.

    Returns
    -------
    - fly_Id : int
        The index of the fly's position in the meas_now array.
    '''

    possible_answers = ["up", "down", "left", "right"]
    val = input('Where is the real fly? (up/down/left/right)\n')

    while val not in possible_answers:
        val = input('Please enter one of the following options: up, down, left, right\n')
    
    if val == 'right':
        fly_Id = 1 if meas_now[0][0] < meas_now[1][0] else 0
    elif val == 'left':
        fly_Id = 0 if meas_now[0][0] < meas_now[1][0] else 1
    elif val == 'up':
        fly_Id = 0 if meas_now[0][1] < meas_now[1][1] else 1
    elif val == 'down':
        fly_Id = 1 if meas_now[0][1] < meas_now[1][1] else 0

    return fly_Id

def determine_fly_robot2(contours):
    '''
    Determines the fly's ID based on the contour with the largest area.

    Parameters
    ----------
    - contours : list of ndarrays
        Contours detected in the frame.

    Returns
    -------
    - fly_id : int
        The index of the contour representing the fly.
    '''
    previous_area = 0
    for i, contour in enumerate(contours):
        area = cv.contourArea(contour)
        if area > previous_area:
            fly_id = i
            previous_area = area
        
    return fly_id


def find_robot_size(robot_contour):
    '''
    Finds the size of the robot by approximating it to a square.

    Parameters
    ----------
    - robot_contour : list of contours
        The contour of the robot.

    Returns
    -------
    - size : tuple, shape (height,width)
        Tuple containing the height and width of the rectangle approximating the robot.
    '''
    #contour_array = np.array(robot_contour, dtype=np.float32)
    rectangle_approx = cv.minAreaRect(robot_contour)
    
    # Get rectangle size
    if rectangle_approx[1][0] > rectangle_approx[1][1]:
        height = rectangle_approx[1][0]
        width = rectangle_approx[1][1]
    else:
        height = rectangle_approx[1][1]
        width = rectangle_approx[1][0]


    size = (width, height)
    return size


def feasible_angles(vertical_walls, horizontal_walls, fly_size, robot_size, space, space_to_wall, meas_now, frame):
    '''
    Finds all feasible angle positions that the robot can go to.

    Parameters
    ----------
    - vertical_walls : list of ndarray
        Array containing the end points of the vertical walls.
    - horizontal_walls : list of ndarray
        Array containing the end points of the horizontal walls.
    - fly_size : float
        The length of the fly's approximation to an ellipse.
    - robot_size : tuple, shape (width, height)
        The size of the robot.
    - space : float
        The space needed between the fly and the robot.
    - space_to_wall : float
        The space needed between the robot and the walls.
    - meas_now : tuple, shape (1,2)
        The center position of the fly.
    - frame : ndarray, shape (height, width, 3)
        The current BGR frame for drawing the radius.

    Returns
    -------
    - feasible_angles : list of int
        List of feasible angles for the robot.
    - space_radius : float
        The radius from the center of the fly that the robot should be at.
    '''
    #compute total radius of space needed from the center of the fly
    vertical_space = (fly_size/2) + space + robot_size[0] 
    space_radius = math.sqrt(vertical_space**2 + (robot_size[1]/2)**2) + space_to_wall
    radius_robot_center = math.sqrt(((fly_size/2) + space + robot_size[0]/2)**2 + (robot_size[1]/2)**2)

    cv.circle(frame, (int(meas_now[0]),int(meas_now[1])), int(space_radius), (255,0,0), 3)

    #find the limits of the walls
    small_limit_x = vertical_walls[0][0][0]
    big_limit_x = vertical_walls[1][0][0]
    if small_limit_x > big_limit_x:
        intermediate = big_limit_x
        big_limit_x = small_limit_x
        small_limit_x = intermediate
    small_limit_y = horizontal_walls[0][0][1]
    big_limit_y = horizontal_walls[1][0][1]
    if small_limit_y > big_limit_y:
        intermediate = big_limit_y
        big_limit_y = small_limit_y
        small_limit_y = intermediate
    
    #determine the feasible angles
    angles = np.arange(0, 360)
    feasible_angles = []
    for j in range(len(angles)):
        #need to add to angle for robot space
        extra_angle = (abs(math.atan2(robot_size[1]/2,vertical_space))*180/math.pi)
        angle_max = (angles[j] + extra_angle)%360
        angle_min = (angles[j] - extra_angle)%360
        if angle_max < angle_min:
            save_min = angle_min
            angle_min = angle_max
            angle_max = save_min

        #compute x and y positions of the robot with respect to all angles
        if angle_max >= 0 and angle_max < 90:
            x_pos_max = meas_now[0] + abs(math.sin(angle_max*math.pi/180)) * space_radius
            y_pos_max = meas_now[1] - abs(math.cos(angle_max*math.pi/180)) * space_radius
        elif angle_max >= 90 and angle_max < 180:
            x_pos_max = meas_now[0] + abs(math.sin((180 - angle_max)*math.pi/180)) * space_radius
            y_pos_max = meas_now[1] + abs(math.cos((180 - angle_max)*math.pi/180)) * space_radius
        elif angle_max >= 180 and angle_max < 270:
            x_pos_max = meas_now[0] - abs(math.sin((angle_max-180)*math.pi/180)) * space_radius
            y_pos_max = meas_now[1] + abs(math.cos((angle_max-180)*math.pi/180)) * space_radius
        elif angle_max >= 270:
            x_pos_max = meas_now[0] - abs(math.sin((360-angle_max)*math.pi/180)) * space_radius
            y_pos_max = meas_now[1] - abs(math.cos((360-angle_max)*math.pi/180)) * space_radius

        if angle_min>= 0 and angle_min < 90:
            x_pos_min = meas_now[0] + abs(math.sin(angle_min*math.pi/180)) * space_radius
            y_pos_min = meas_now[1] - abs(math.cos(angle_min*math.pi/180)) * space_radius
        elif angle_min >= 90 and angle_min < 180:
            x_pos_min = meas_now[0] + abs(math.sin((180 - angle_min)*math.pi/180)) * space_radius
            y_pos_min = meas_now[1] + abs(math.cos((180 - angle_min)*math.pi/180)) * space_radius
        elif angle_min >= 180 and angle_min < 270:
            x_pos_min = meas_now[0] - abs(math.sin((angle_min-180)*math.pi/180)) * space_radius
            y_pos_min = meas_now[1] + abs(math.cos((angle_min-180)*math.pi/180)) * space_radius
        elif angle_min >= 270:
            x_pos_min = meas_now[0] - abs(math.sin((360-angle_min)*math.pi/180)) * space_radius
            y_pos_min = meas_now[1] - abs(math.cos((360-angle_min)*math.pi/180)) * space_radius

        #determine if the position of the robot would fit within the frame's walls
        if x_pos_min >= small_limit_x and x_pos_min <= big_limit_x and y_pos_min >= small_limit_y and y_pos_min <= big_limit_y and x_pos_max >= small_limit_x and x_pos_max <= big_limit_x and y_pos_max >= small_limit_y and y_pos_max <= big_limit_y:
            feasible_angles.append(angles[j]) 
    
    return feasible_angles, radius_robot_center


def define_StartEnd_points(feasible_angles, meas_now,radius_robot_center):
    '''
    Determines the starting and ending positions of the robot.

    Parameters
    ----------
    - space : float
        The space between the robot and the fly.
    - fly_size : float
        The length of the fly's approximation to an ellipse.
    - robot_size : tuple, shape (width, height)
        The size of the robot.
    - feasible_angles : list of int
        List of all feasible angle positions that the robot can go to.
    - meas_now : tuple, shape (1,2)
        The center position of the fly.

    Returns
    -------
    - start_point_x : ndarray
        Array of x-coordinates for the starting positions of the robot.
    - start_point_y : ndarray
        Array of y-coordinates for the starting positions of the robot.
    - end_point_x : ndarray
        Array of x-coordinates for the ending positions of the robot.
    - end_point_y : ndarray
        Array of y-coordinates for the ending positions of the robot.
    '''

    angle_start = 0
    angle_finish = 0

    # Find longest range feasible in feasible_angles list:
    current_range = [feasible_angles[0]]
    longest = []
    for i in range(1,len(feasible_angles)):
        if feasible_angles[i] == (current_range[-1] + 1):
            current_range.append(feasible_angles[i])
        else:
            if len(current_range) > len(longest):
                longest = current_range
                angle_start = current_range[0]
                angle_finish = current_range[-1]
            current_range = [feasible_angles[i]]
        
    if len(current_range) > len(longest):
        longest = current_range
        angle_start = current_range[0]
        angle_finish = current_range[-1]
    
    # Check if there is a wrap around 
    if longest[-1] == 359 and feasible_angles[0] == 0 and (0 not in longest):
        longest.append(0)
        for i in range(1,len(feasible_angles)):
            if feasible_angles[i] == (longest[-1] + 1):
                longest.append(feasible_angles[i])
            else:
               break
        angle_start = longest[0]
        angle_finish = longest[-1]
    
    if longest[0] == 0 and feasible_angles[-1] == 359 and (359 not in longest):
        longest.insert(0,359)
        for i in range(len(feasible_angles)-2,-1,-1):
            if feasible_angles[i] == (longest[0] - 1):
                longest.insert(0,feasible_angles[i])
            else:
                break
        angle_start = longest[0]
        angle_finish = longest[-1]

    # Convert the angles to be with respect to horizontal axis
    start_hori = 0
    finish_hori = 0
 
    if angle_start >= 0 and angle_start< 90:
        start_hori = 90 - angle_start
        start_point_x = meas_now[0] + (radius_robot_center)*abs(math.sin(abs(angle_start)*math.pi/180))
        start_point_y = meas_now[1] - (radius_robot_center)*abs(math.cos(abs(angle_start)*math.pi/180))

    elif angle_start >= 90 and angle_start < 180:
        start_hori= (450-angle_start)%360
        start_point_x = meas_now[0] + (radius_robot_center)*abs(math.cos(abs(angle_start-90)*math.pi/180))
        start_point_y = meas_now[1] + (radius_robot_center)*abs(math.sin(abs(angle_start-90)*math.pi/180))

    elif angle_start >= 180 and angle_start < 270:
        start_hori = (450-angle_start)%360
        start_point_x = meas_now[0] -  (radius_robot_center)*abs(math.sin(abs(angle_start-180)*math.pi/180))
        start_point_y = meas_now[1] +  (radius_robot_center)*abs(math.cos(abs(angle_start-180)*math.pi/180))
    elif angle_start >= 270:
        start_hori = (450-angle_start)%360
        start_point_x  = meas_now[0] - (radius_robot_center)*abs(math.sin(abs(360-angle_start)*math.pi/180))
        start_point_y = meas_now[1] - (radius_robot_center)*abs(math.cos(abs(360-angle_start)*math.pi/180))


    if angle_finish >= 0 and angle_finish < 90:
        finish_hori = 90 - angle_finish
        end_point_x = meas_now[0] + (radius_robot_center)*abs(math.sin(abs(angle_finish)*math.pi/180))
        end_point_y = meas_now[1] - (radius_robot_center)*abs(math.cos(abs(angle_finish)*math.pi/180))
    elif angle_finish >= 90 and angle_finish < 180:
        finish_hori = (450-angle_finish)%360
        end_point_x = meas_now[0] + (radius_robot_center)*abs(math.cos(abs(angle_finish-90)*math.pi/180))
        end_point_y = meas_now[1] + (radius_robot_center)*abs(math.sin(abs(angle_finish-90)*math.pi/180))
    elif angle_finish >= 180 and angle_finish < 270:
        finish_hori = (450-angle_finish)%360
        end_point_x = meas_now[0] - (radius_robot_center)* abs(math.sin(abs(angle_finish-180)*math.pi/180))
        end_point_y = meas_now[1] + (radius_robot_center)* abs(math.cos(abs(angle_finish-180)*math.pi/180))
    elif angle_finish >= 270:
        finish_hori = (450-angle_finish)%360
        end_point_x = meas_now[0] - (radius_robot_center)* abs(math.sin(abs(360-angle_finish)*math.pi/180))
        end_point_y  = meas_now[1] - (radius_robot_center)* abs(math.cos(abs(360-angle_finish)*math.pi/180))

    return start_point_x, start_point_y, end_point_x, end_point_y, start_hori, finish_hori


def double_check(previous_half_intensity, current_half_intensity, corrected_times, number):
    '''
    Double checks that the determined fly's head position (the angle of the fly) is correct.
    If the intensity of the 10 previous head positions differs on average from the current one,
    and if the position hasn't already been corrected at least 5 times, then we change the prediction.

    Parameters
    ----------
    - previous_half_intensity : ndarray, shape (10,)
        The intensity of the 10 previous positions of the highest intensity.
        1 -> top half has the highest intensity, and 2 -> bottom half has the highest intensity.
    - current_half_intensity : int
        Current predicted half that has the highest intensity.
    - corrected_times : int
        Number of times we have corrected the head's position.
    - number : int
        The number of times we want to correct the position in a row

    Returns
    -------
    - possible : bool
        True if the predicted head position is possible, False if we need to change it.
    - corrected_times : int
        Updated number of times we have corrected the head's position.
    '''

    # Compute the mean of the previous values
    mean_half = np.mean(previous_half_intensity)
    possible = True

    if mean_half > 1.5 :
        int_half = 2
    else :
        int_half = 1

    # If the current value is different from the previous ones and we haven't aloready corrected it 15 times
    if int_half != current_half_intensity and corrected_times <= number :
        corrected_times = corrected_times + 1
        possible = False
    elif int_half == current_half_intensity:
        # Reset the correct times variable
        corrected_times = 0

    return possible, corrected_times


def add_to_array(array, value_to_add, length):
    '''
    Adds a value to an array while maintaining its size to the specified length:
        - Discards the first value of the array if it exceeds the specified length.
        - Shifts the array to the left.
        - Appends one value to the end of the array.

    Parameters:
    - array : list
        The array to which the value will be added.
    - value_to_add : any 
        The value to add to the array.
    - length : int
        The maximum length of the array.

    Returns:
    - new_array : list
        An array of at most the specified length with the new value added.
    '''

    # Check if the array length is less than the specified maximum length
    if len(array) < length:
        array.append(value_to_add)  # Append the value to the array
        new_array = array  # Assign the new array
    else:
        # If the array length exceeds the specified maximum length
        # Remove the first value and append the new value
        new_array = array[1:] + [value_to_add]

    return new_array


def determine_if_possible(desired_start, desired_end, feasible_angles, fly_angle):
    '''
    Determines if the robot can perform the trajectory from the desired start to end angles based on the feasible angles array.

    Parameters:
    - desired_start : int
        The desired start angle with respect to the fly's head position.
    - desired_end : int
        The desired end angle with respect to the fly's head position.
    - feasible_angles : list of int
        Array containing all feasible angles for the robot.
    - fly_angle : float
        The angle of the fly with respect to the vertical axis [0, 180] and [0, -180].

    Returns:
    - possible : bool
        True if it is possible for the robot to perform the trajectory from desired start to end angles; False otherwise.
    '''

    # Convert fly angle to be with respect to vertical axis
    if fly_angle < 0:
        fly_angle = fly_angle + 360
    desired_start_vertical = (desired_start + fly_angle) % 360
    desired_end_vertical = (desired_end + fly_angle) % 360

    possible = True
    desired_trajectory_array = np.arange(desired_start_vertical, desired_end_vertical + 1)
    # Check if all of the angles that will be used are in the feasible_angles array
    for angle in desired_trajectory_array:
        if angle not in feasible_angles:
            possible = False
            break

    return possible
        

def predict_facing(cnn, svm, both, pca_reloaded, modelUp_or_Down, model_CNN, final_ML):
    '''
    Predicts the direction the robot is facing using either SVM, CNN, or both.

    Parameters:
    - cnn : bool
        If True, uses the CNN model for prediction.
    - svm : bool
        If True, uses the SVM model for prediction.
    - both : bool
        If True, uses both SVM and CNN models for prediction.
    - pca_reloaded : object
        Pre-trained PCA model for dimensionality reduction.
    - modelUp_or_Down : object
        Pre-trained SVM model for predicting up or down direction.
    - model_CNN : object
        Pre-trained CNN model for predicting direction.
    - final_ML : ndarray
        Input data for prediction.

    Returns:
    - predicted_way : int
        Predicted direction (1 for stomach, 2 for back).
    '''

    if both:
        # Predict with SVM
        pca_image_svm = prepare_for_ML(final_ML, pca_reloaded, False, True)
        decision_values_svm = modelUp_or_Down.decision_function(pca_image_svm)
        probabilities_svm = 1 / (1 + np.exp(-decision_values_svm))
        confidence_svm = np.max(probabilities_svm, axis=0)  # Get the max pseudo-probability for each sample
        predicted_way_svm = np.argmax(probabilities_svm, axis=0)

        # Predict with CNN
        pca_image_cnn = prepare_for_ML(final_ML, pca_reloaded, True, False)
        with torch.no_grad():
            outputs_cnn = model_CNN(pca_image_cnn)
            probabilities_cnn = F.softmax(outputs_cnn, dim=1)
            confidence_cnn, predicted_class_cnn = torch.max(probabilities_cnn, 1)
            if predicted_class_cnn[0].item() == 0:
                predicted_way_cnn = 2
            else:
                predicted_way_cnn = predicted_class_cnn[0].item()
        
        # Choose prediction with higher confidence
        if confidence_svm > confidence_cnn:
            predicted_way = predicted_way_svm
        else:
            predicted_way = predicted_way_cnn
    elif cnn:
        # Predict with CNN
        pca_image_cnn = prepare_for_ML(final_ML, pca_reloaded, True, False)
        with torch.no_grad():
            outputs_cnn = model_CNN(pca_image_cnn)
            probabilities_cnn = F.softmax(outputs_cnn, dim=1)
            _, predicted_class_cnn = torch.max(probabilities_cnn, 1)
            if predicted_class_cnn[0].item() == 0:
                predicted_way = 2
            else:
                predicted_way = predicted_class_cnn[0].item()
    elif svm:
        # Predict with SVM
        pca_image_svm = prepare_for_ML(final_ML, pca_reloaded, False, True)
        predicted_way = modelUp_or_Down.predict(pca_image_svm)[0]
        print(predicted_way)

    return predicted_way

def draw_end_pos(start_point_x, start_point_y, end_point_x, end_point_y, meas_now, fly_id, robot_size, frame_color_output):
    '''
    Draws the starting and ending positions of the robot on the frame.

    Parameters:
    - start_point_x : float
        X-coordinate of the starting position of the robot.
    - start_point_y : float
        Y-coordinate of the starting position of the robot.
    - end_point_x : float
        X-coordinate of the ending position of the robot.
    - end_point_y : float
        Y-coordinate of the ending position of the robot.
    - meas_now : list of tuples
        List of tuples containing the current positions of the flies.
    - fly_id : int
        Index of the fly for which the position is being determined.
    - robot_size : tuple
        Tuple containing the width and height of the robot.
    - frame_color_output : ndarray
        Frame to draw the robot's position on.

    Returns:
    - None
    '''

    # Draw circles at the starting and ending positions
    cv.circle(frame_color_output, (int(start_point_x), int(start_point_y)), 3, (0, 255, 0), -1)
    cv.circle(frame_color_output, (int(end_point_x), int(end_point_y)), 3, (0, 255, 0), -1)

    # Compute rotation angles for the robot
    delta_y1 = start_point_y - meas_now[fly_id][1]
    delta_x1 = start_point_x - meas_now[fly_id][0]
    rotation_angle1 = math.atan2(delta_x1, delta_y1) * 180 / math.pi
    if rotation_angle1 < 0:
        rotation_angle1 += 360
    rotation_angle1 += 90

    delta_y2 = end_point_y - meas_now[fly_id][1]
    delta_x2 = end_point_x - meas_now[fly_id][0]
    rotation_angle2 = math.atan2(delta_x2, delta_y2) * 180 / math.pi
    if rotation_angle2 < 0:
        rotation_angle2 += 360
    rotation_angle2 += 90

    # Get rotation matrices
    rotation_matrix1 = cv.getRotationMatrix2D((int(start_point_x), int(start_point_y)), int(rotation_angle1), 1)
    rotation_matrix2 = cv.getRotationMatrix2D((int(end_point_x), int(end_point_y)), int(rotation_angle2), 1)

    # Define corner points of the rectangle
    width = robot_size[0]
    height = robot_size[1]
    rect_points = np.array([[-width / 2 + start_point_x, -height / 2 + start_point_y],
                            [width / 2 + start_point_x, -height / 2 + start_point_y],
                            [width / 2 + start_point_x, height / 2 + start_point_y],
                            [-width / 2 + start_point_x, height / 2 + start_point_y]], dtype=np.float32)

    rect_points2 = np.array([[-width / 2 + end_point_x, -height / 2 + end_point_y],
                             [width / 2 + end_point_x, -height / 2 + end_point_y],
                             [width / 2 + end_point_x, height / 2 + end_point_y],
                             [-width / 2 + end_point_x, height / 2 + end_point_y]], dtype=np.float32)

    # Rotate the rectangle points
    rotated_rect_points = cv.transform(np.array([rect_points]), rotation_matrix1)[0]
    rotated_rect_points2 = cv.transform(np.array([rect_points2]), rotation_matrix2)[0]

    # Convert points to integer
    rotated_rect_points = rotated_rect_points.astype(np.int32)
    rotated_rect_points2 = rotated_rect_points2.astype(np.int32)

    # Draw the rotated rectangle
    cv.polylines(frame_color_output, [rotated_rect_points], isClosed=True, color=(255, 0, 0), thickness=3)
    cv.polylines(frame_color_output, [rotated_rect_points2], isClosed=True, color=(255, 0, 0), thickness=3)

    return None


def detect_contours(thresh, meas_now,frame_no, frame_gray,frame):
    """
    This function detects contours, thresholds them based on area and keeps only the 2 biggest (fly and robot)
    Parameters
    ----------
    thresh: ndarray, shape(n_rows, n_cols, 1)
        binarised(0,255) image
    meas_now: array_like, dtype=float
        individual's location on current frame
        
    Returns
    -------
    contours: list
        a list of all detected contours that pass the area based threhold criterion
    meas_last: array_like, dtype=float
        individual's location on previous frame
    meas_now: array_like, dtype=float
        individual's location on current frame
    """
    
    # Detect contours and draw them based on specified area thresholds
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    #find area and sort it to keep only the 2 biggest contours
    area_array = []
    for contour in contours:
        area = cv.contourArea(contour)
        area_array.append(area)
    area_array = np.array(area_array)
    sorted_indices = np.argsort(area_array)
    sorted_contours = [contours[i] for i in sorted_indices]

    if len(sorted_contours) == 0:
        cv.imshow('thresh', thresh)
        cv.waitKey(0)
        cv.imshow('no walls', frame_no)
        cv.waitKey(0)
        cv.imshow('gray', frame_gray)
        cv.waitKey(0)
        cv.imshow('frame', frame)
        cv.waitKey(0)

    #keep only the 2 biggest contours
    kept_contours = sorted_contours[-2:]

    meanArea = np.mean(area_array)
    contours = [contour for contour in kept_contours if cv.contourArea(contour) > 0.5*meanArea]

    i = 0
    meas_last = meas_now.copy()
    del meas_now[:]

    while i < len(contours):
        M = cv.moments(contours[i])
        if M['m00'] != 0:
            cx = M['m10']/M['m00']
            cy = M['m01']/M['m00']
        else:
            cx = 0
            cy = 0
        meas_now.append([cx,cy])
        i += 1

    return contours, meas_last, meas_now


def hungarian_algorithm(meas_last, meas_now):
    """
    The hungarian algorithm is a combinatorial optimisation algorithm used
    to solve assignment problems. Here, we use the algorithm to reduce noise
    due to ripples and to maintain individual identity. This is accomplished
    by minimising a cost function; in this case, euclidean distances between 
    points measured in previous and current step. The algorithm here is written
    to be flexible as the number of contours detected between successive frames
    changes. However, an error will be returned if zero contours are detected.
   
    Parameters
    ----------
    meas_last: array_like, dtype=float
        individual's location on previous frame
    meas_now: array_like, dtype=float
        individual's location on current frame
        
    Returns
    -------
    row_ind: array, dtype=int64
        individual identites arranged according to input ``meas_last``
    col_ind: array, dtype=int64
        individual identities rearranged based on matching locations from 
        ``meas_last`` to ``meas_now`` by minimising the cost function
    """
    meas_last = np.array(meas_last)
    meas_now = np.array(meas_now)
    if meas_now.shape != meas_last.shape:
        if meas_now.shape[0] < meas_last.shape[0]:
            while meas_now.shape[0] != meas_last.shape[0]:
                meas_last = np.delete(meas_last, meas_last.shape[0]-1, 0)
        else:
            result = np.zeros(meas_now.shape)
            result[:meas_last.shape[0],:meas_last.shape[1]] = meas_last
            meas_last = result

    meas_last = list(meas_last)
    meas_now = list(meas_now)
    cost = cdist(meas_last, meas_now)
    row_ind, col_ind = linear_sum_assignment(cost)
    
    return row_ind, col_ind


def reorder_and_draw(final, colours, n_inds, col_ind, contours, fr_no, meas_now, mot=True):
    """
    This function reorders the measurements in the current frame to match
    identity from previous frame. This is done by using the results of the
    hungarian algorithm from the array col_inds.
    
    Parameters
    ----------
    final: ndarray, shape(n_rows, n_cols, 3)
        final output image composed of the input frame with object contours 
        and centroids overlaid on it
    colours: list, tuple
        list of tuples that represent colours used to assign individual identities
    n_inds: int
        total number of individuals being tracked
    col_ind: array, dtype=int64
        individual identities rearranged based on matching locations from 
        ``meas_last`` to ``meas_now`` by minimising the cost function
    meas_now: array_like, dtype=float
        individual's location on current frame
    df: pandas.core.frame.DataFrame
        this dataframe holds tracked coordinates i.e. the tracking results
    mot: bool
        this boolean determines if we apply the alogrithm to a multi-object
        tracking problem
        
    Returns
    -------
    contours: list of contours reordered
    """
    # Reorder contours based on results of the hungarian algorithm
    font = cv.FONT_HERSHEY_SCRIPT_SIMPLEX
    equal = np.array_equal(col_ind, list(range(len(col_ind))))
    if equal == False:
        current_ids = col_ind.copy()
        reordered = [i[0] for i in sorted(enumerate(current_ids), key=lambda x:x[1])]
        meas_now = [x for (y,x) in sorted(zip(reordered,meas_now))]
        contours = [contours[i] for i in reordered]

    # Draw centroids
    if mot == False:
        for i in range(len(meas_now)):
            if colours[i%4] == (0,0,255):
                cv.circle(final, tuple([int(x) for x in meas_now[i]]), 5, colours[i%4], -1, cv.LINE_AA)
    else:
        for i in range(n_inds):
            cv.putText(final,str(i),tuple([int(x) for x in meas_now[i]]),font,2,colours[i%n_inds],2,cv.LINE_AA)
            cv.circle(final, tuple([int(x) for x in meas_now[i]]), 5, colours[i%n_inds], -1, cv.LINE_AA)
    
    # add frame number
    cv.putText(final, str(int(fr_no)), (5,30), font, 1, (255,255,255), 2)

    return contours, meas_now


def approximate_to_ellipse(frame, rotated_mask1, rotated_mask2, rotated_mask, rotated_frame):
    """
    Approximates the fly's contour in the frame to an ellipse and returns only the part of the frame that contains the ellipse.

    Parameters
    ----------
    - frame : ndarray (n_rows,n_cols)
        Grayscale frame for analysis.
    - rotated_mask1 : ndarray
        Translated, rotated, and cut (top part) mask of the approximation to an ellipse of the fly's contour.
    - rotated_mask2 : ndarray
        Translated, rotated, and cut (bottom part) mask of the approximation to an ellipse of the fly's contour.
    - rotated_mask : ndarray
        Translated and rotated mask of the fly's approximation to an ellipse without being cut.
    - rotated_frame : ndarray
        Translated and rotated frame of the fly's approximation to an ellipse without being cut.
        
    Returns
    -------
    final_frame : ndarray
        Frame cut to only the ellipse approximation of the fly.
    """

    #Apply mask to image
    masked_image1 = cv.bitwise_and(rotated_frame, rotated_mask1)
    masked_image2 = cv.bitwise_and(rotated_frame, rotated_mask2)
    
    # Calculate the average pixel intensity for each half
    average_intensity_half1 = np.mean(masked_image1)
    average_intensity_half2 = np.mean(masked_image2)

    #fit the ellipse to the frame
    if average_intensity_half1 > average_intensity_half2:
        final_frame = cv.bitwise_and(rotated_frame, rotated_mask)
    else:
        rotation_matrix180 = cv.getRotationMatrix2D((frame.shape[1]/2, frame.shape[0]/2), 180, 1.0)
        rotated_mask180 = cv.warpAffine(rotated_mask, rotation_matrix180, (frame.shape[1], frame.shape[0]))
        rotated_frame180 = cv.warpAffine(rotated_frame, rotation_matrix180, (frame.shape[1], frame.shape[0]))
        final_frame = cv.bitwise_and(rotated_frame180,rotated_mask180)


    #crop image
    final_frame = final_frame[332:692,402:622]

    return final_frame