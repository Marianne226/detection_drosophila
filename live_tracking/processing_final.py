import sys
import cv2
import numpy as np
import torch
from torchvision import models, transforms
import torch.nn as nn
import csv
from joblib import load
from imageio import get_writer
from pipeline_final import *



# Load the CNN model
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # Adjust to match the original model's output units
model.load_state_dict(torch.load('fly_CNN_model3_4_06Best.pth'))
model.eval()
model_CNN = model

# Load the PCA algorithm
pca_reloaded = load('pca_model_added.sav')

# Load the SVM algorithm
modelUp_or_Down = load('finalized_model_added.sav')

exp = Experiment()
count = 0

# If we want to run the tracking without the GUI : 
'''
# To save the processed video 
writer = get_writer('output.mp4', fps=80)

while True:
    # Read the image from the camera
    img_bytes = sys.stdin.buffer.read(1024 * 1024)
    img = np.frombuffer(img_bytes, dtype=np.uint8)

    try:
        img = img.reshape((1024, 1024))
    except ValueError:
        break

    # Process the frame
    final, radius, delta_x_robot, delta_y_robot, start_ang_h, finish_ang_h, fly_pos = exp.process_video(img, count, pca_reloaded, modelUp_or_Down,model_CNN, cnn=True, svm=False, both=False)

    writer.append_data(final)

    cv2.imshow("Final image", final)
    cv2.waitKey(1)
    count += 1

writer.close()

# Save the time metrics to a csv file
with open('tracking_metrics.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['count', 'time', 'total_elapsed_time','preprocess_elapsed_time','correction_elapsed_time','detection_elapsed_time','reorder_elapsed_time','fly_head_elapsed_time', 'facing_elapsed_time', 'feasible_pos_elapsed_time', 'draw_magnet_elapsed_time'])
    csvwriter.writerows(exp.metrics)'''

# Functions to use with GUI : 
def image_proc(img,count):
    '''
    Function called by GUI to process the image
    Parameters:
    - img : ndarray, shape (height, width, 3)
        The current BGR frame to be processed
    - count : int
       Number of the frame
    Returns:
    - final : ndarray, shape (height, width, 3)
        Processed frame with drawings
    - radius : int
        Radius that the magnet should move at around the fly
    - delta_x_robot : int
        X position of the robot with respect to the magnet
    - delta_y_robot : int
        Y position of the robot with respect to the magnet
    - start_ang_h : int
        Maximum possible starting angle of the magnet in trigonometric direction
    - finish_ang_h : int
        Maximum possible finishing angle of the magnet in trigonometric direction
    - fly_pos : (tuple)
        Position of the fly
    '''
    global exp
    global writer
    
    final, radius, delta_x_robot, delta_y_robot, start_ang_h, finish_ang_h, fly_pos = exp.process_video(img, count, pca_reloaded, modelUp_or_Down,model_CNN, cnn=True, svm=False, both=False)

    '''
    # To save the processed video 
    writer.append_data(final)

    # Save the time metrics to a csv file /!\ change the 400 to the desired processed frame number
    if count == 400:
        with open('tracking_metrics5.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['count', 'time', 'total_elapsed_time','preprocess_elapsed_time','correction_elapsed_time','detection_elapsed_time','reorder_elapsed_time','fly_head_elapsed_time', 'facing_elapsed_time', 'feasible_pos_elapsed_time', 'draw_magnet_elapsed_time'])
            csvwriter.writerows(exp.metrics)'''

    return final, radius, delta_x_robot, delta_y_robot, start_ang_h, finish_ang_h

def call_reset():
    '''
    Function used by GUI to call the reset function of the tracking algorithm
    This will reset the IDs of the fly and the magnet
    '''
    global exp

    exp.reset_call()

    return

def switch_command():
    '''
    Function used by GUI to call the switching function of the tracking algorithm
    This will switch the IDs of the fly and the magnet
    '''
    global exp

    exp.switch_command()

    return
