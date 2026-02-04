import numpy as np
import cv2 as cv

def thresholdImage(frame):
        '''
        Performs binary thresholding of the image using Otsu's algorithm
        Input: 
            - frame: grayscale frame with background subtracted
        Output:
            - thresh: binary thresholded image of the flies such that the flies are white and the rest of the image 
                      is black
        '''
        # Blur to Clean Noise 
        noiseCleaned = cv.blur(frame,(25,25))

        ret, thresh = cv.threshold(noiseCleaned,50,255,cv.THRESH_BINARY)
        return thresh

def preprocessing(frame):
    '''
    Finds the number of flies in the experiment as well as their mean area in squared pixels.

    Parameters
    ----------
    frame : ndarray
        BGR frame of the first n seconds of the experiment.

    Returns
    -------
    number : int
        Number of flies in the experiment.
    meanArea : float
        Mean area of the flies in squared pixels.
    '''
    # Convert the frame to grayscale
    gray_frame = cv.cvtColor(frame.copy(), cv.COLOR_BGR2GRAY)

    # Copy the frame to noBKG
    noBKG = gray_frame.copy()

    # Threshold the image to segment the flies
    thresholded = thresholdImage(noBKG)

    # Find contours in the thresholded image
    contours, _ = cv.findContours(thresholded, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Calculate the area of each contour
    area_array = [cv.contourArea(contour) for contour in contours]
    area_array = np.array(area_array)

    # Sort contours by area
    sorted_indices = np.argsort(area_array)
    sorted_contours = [contours[i] for i in sorted_indices]

    # Keep only the 2 biggest contours
    kept_contours = sorted_contours[-2:]

    # Calculate the mean area of all contours
    meanArea = np.mean(area_array)

    # Filter out contours that are smaller than 80% of the mean area
    contoursToKeep = [contour for contour in kept_contours if cv.contourArea(contour) > 0.8 * meanArea]

    # The number of contours to keep is the number of flies
    number = len(contoursToKeep)

    return number, meanArea


def detect_and_draw_contours_new(frame, thresh, meanArea, meas_now):
    """
    Detects contours, filters them based on area, and draws them on the frame.

    Parameters
    ----------
    frame : ndarray, shape (n_rows, n_cols, 3)
        Source image containing all three color channels.
    thresh : ndarray, shape (n_rows, n_cols, 1)
        Binarized (0, 255) image.
    meanArea : int
        Mean area of the contours.
    meas_now : list of lists
        Individual's location on the current frame.

    Returns
    -------
    final : ndarray, shape (n_rows, n_cols, 3)
        Final output image with object contours and centroids overlaid.
    contours : list
        List of all detected contours that pass the area-based threshold criterion.
    meas_last : list of lists
        Individual's location on the previous frame.
    meas_now : list of lists
        Updated individual's location on the current frame.
    """

    # Detect contours in the thresholded image
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    # Calculate the area of each contour and sort them
    area_array = [cv.contourArea(contour) for contour in contours]
    area_array = np.array(area_array)
    sorted_indices = np.argsort(area_array)
    sorted_contours = [contours[i] for i in sorted_indices]

    # Keep only the two largest contours
    kept_contours = sorted_contours[-2:]
    
    # Filter contours based on area threshold (70% of mean area)
    contours = [contour for contour in kept_contours if cv.contourArea(contour) > 0.7 * meanArea]

    # Copy the original frame to draw contours on
    final = frame.copy()

    # Store the current measurements and prepare to update them
    meas_last = meas_now.copy()
    meas_now.clear()

    # Iterate through the filtered contours
    for i, contour in enumerate(contours):
        # Draw each contour on the final image
        cv.drawContours(final, contours, i, (0, 0, 255), 1)

        # Calculate the centroid of the contour
        M = cv.moments(contour)
        if M['m00'] != 0:
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
        else:
            cx = 0
            cy = 0
        
        # Update the current measurements with the centroid coordinates
        meas_now.append([cx, cy])
    
    return final, contours, meas_last, meas_now


def correctDetection(frame, num):
    """
    Corrects the detection of flies in a frame by eroding and dilating contours.

    Parameters
    ----------
    frame : ndarray, shape (n_rows, n_cols)
        Source grayscale image.
    num : int
        Expected number of flies.

    Returns
    -------
    dilated_contours : list
        List of dilated contours.
    meanArea : float
        Mean area of the detected contours.
    corrected : bool
        True if the number of contours matches the expected number after correction, False otherwise.
    """
    
    corrected = False
    
    # Erode the frame to remove noise and small objects
    kernel1 = np.ones((50, 50), np.uint8)
    eroded = cv.erode(frame, kernel1, iterations=1)
    
    # Find contours in the eroded image
    contours, _ = cv.findContours(eroded, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    dilated_contours = []

    # Dilate each contour and find new contours in the dilated mask
    for contour in contours:
        mask = np.zeros_like(frame)
        cv.drawContours(mask, [contour], 0, 255, -1)  # Draw the contour as a filled white shape
        dilated_mask = cv.dilate(mask, np.ones((50, 50), np.uint8), iterations=1)  # Dilate the mask
        contour_dilated, _ = cv.findContours(dilated_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        for contour2 in contour_dilated:
            dilated_contours.append(contour2)

    # Calculate the area of each contour
    areas = [cv.contourArea(contour) for contour in contours]
    meanArea = np.mean(np.array(areas))

    # Filter contours based on 80% of the mean area
    contoursToKeep = [contour for contour in contours if cv.contourArea(contour) > 0.8 * meanArea]

    # Check if the number of filtered contours matches the expected number
    if len(contoursToKeep) == num:
        corrected = True

    return dilated_contours, meanArea, corrected

def reformatted(last, now):
    """
    Reformats the `now` list to match the order of the `last` list based on the minimum Euclidean distance.
    
    Parameters
    ----------
    last : list of lists
        List containing the positions from the last frame.
    now : list of lists
        List containing the positions from the current frame.

    Returns
    -------
    now_reformatted : list of lists
        List containing the positions from the current frame, reordered to match the order of the last frame.
    """
    
    lstIDS = []

    # Iterate over each position in the last frame
    for i in range(len(last)):
        # Calculate the Euclidean distance between the current position in the last frame 
        # and all positions in the current frame
        distList = [np.linalg.norm(np.array(last[i]) - np.array(now[j])) for j in range(len(now))]
        
        # Find the index of the current position in the current frame that has the minimum distance
        lstIDS.append(distList.index(min(distList)))

    # Reorder the current positions to match the order of the last positions
    now_reformatted = [now[i] for i in lstIDS]

    return now_reformatted

def approximate_to_ellipse(contour, frame):
    """
    Approximates a contour to an ellipse, creates masks for the ellipse halves, 
    and returns the cropped image of the rotated frame based on the higher average intensity half.

    Parameters
    ----------
    contour : ndarray
        Contour of the object to be approximated to an ellipse.
    frame : ndarray
        The source image containing the object.

    Returns
    -------
    final_frame : ndarray
        Cropped image of the rotated frame based on the higher average intensity half of the ellipse.
    """
    
    # Approximate the contour to an ellipse
    ellipse = cv.fitEllipse(contour)

    # Create an empty mask
    mask = np.zeros_like(frame)

    # Draw the ellipse on the mask
    cv.ellipse(mask, ellipse, (255, 255, 255), -1)

    # Extract the parameters of the ellipse
    (x, y), (MA, ma), angle_init = ellipse  # Coordinates of center (x, y) and major/minor axes length
    center = (int(x), int(y))

    # Adjust rotation angle if necessary
    if angle_init > 90:
        angle_rot = angle_init - 180
        angle = 180 - angle_init
    else:
        angle_rot = angle_init
        angle = angle_init

    # Create masks for each half of the ellipse
    half1_mask = np.zeros_like(frame)
    half2_mask = np.zeros_like(frame)
    cv.ellipse(half1_mask, ellipse, (255, 255, 255), -1)
    cv.ellipse(half2_mask, ellipse, (255, 255, 255), -1)

    # Center the image and masks
    translation_matrix = np.float32([[1, 0, (frame.shape[1] / 2 - x)], [0, 1, (frame.shape[0] / 2 - y)]])
    translated_frame = cv.warpAffine(frame, translation_matrix, (frame.shape[1], frame.shape[0]))
    translated_mask1 = cv.warpAffine(half1_mask, translation_matrix, (frame.shape[1], frame.shape[0]))
    translated_mask2 = cv.warpAffine(half2_mask, translation_matrix, (frame.shape[1], frame.shape[0]))

    # Rotate the image and mask to cut horizontally
    rotation_matrix = cv.getRotationMatrix2D((frame.shape[1] / 2, frame.shape[0] / 2), angle_rot, 1.0)
    rotated_mask1 = cv.warpAffine(translated_mask1, rotation_matrix, (frame.shape[1], frame.shape[0]))
    rotated_mask = rotated_mask1.copy()
    rotated_mask2 = cv.warpAffine(translated_mask2, rotation_matrix, (frame.shape[1], frame.shape[0]))

    rotated_frame = cv.warpAffine(translated_frame, rotation_matrix, (frame.shape[1], frame.shape[0]))

    # Mask each half of the rotated frame
    rotated_mask1[int((frame.shape[0] / 2 - ma / 4)):, :] = 0
    rotated_mask2[:int((frame.shape[0] / 2 + ma / 4)), :] = 0

    # Apply mask to image
    masked_image1 = cv.bitwise_and(rotated_frame, rotated_mask1)
    masked_image2 = cv.bitwise_and(rotated_frame, rotated_mask2)

    # Calculate the average pixel intensity for each half
    average_intensity_half1 = np.mean(masked_image1)
    average_intensity_half2 = np.mean(masked_image2)

    # Select the final frame based on the higher average intensity half
    if average_intensity_half1 > average_intensity_half2:
        final_frame = cv.bitwise_and(rotated_frame, rotated_mask)
    else:
        rotation_matrix180 = cv.getRotationMatrix2D((frame.shape[1] / 2, frame.shape[0] / 2), 180, 1.0)
        rotated_mask180 = cv.warpAffine(rotated_mask, rotation_matrix180, (frame.shape[1], frame.shape[0]))
        rotated_frame180 = cv.warpAffine(rotated_frame, rotation_matrix180, (frame.shape[1], frame.shape[0]))
        final_frame = cv.bitwise_and(rotated_frame180, rotated_mask180)

    # Crop image to specific region
    final_frame = final_frame[332:692, 402:622]

    return final_frame


def hide_walls(frame):
    """
    Masks the walls of the frame by creating a blank image with a white rectangle in the center,
    and then applies this mask to the frame to hide the borders.

    Parameters
    ----------
    frame : ndarray
        The source image from which the walls (borders) need to be hidden.

    Returns
    -------
    no_walls : ndarray
        The frame with the borders hidden.
    """
    
    # Create a blank image with the same dimensions as the frame
    blank = np.zeros_like(frame)

    # Define the dimensions of the border
    border_width = 25

    # Create a white rectangle in the center of the blank image
    cv.rectangle(blank, (border_width, border_width),
                 (frame.shape[1] - border_width, frame.shape[0] - border_width), (255, 255, 255), -1)
    
    # Apply the mask to the frame to hide the walls
    no_walls = cv.bitwise_and(frame, blank)
    
    return no_walls