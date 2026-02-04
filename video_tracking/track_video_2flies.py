import sys
import numpy as np
import torch
from torchvision import models
import torch.nn as nn
from joblib import load
from pipeline_video_2flies import *
#also works with pipeline_video_magnet --> will assume one fly is the robot and the other is the magnet

# Load the CNN model
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # Adjust to match the original model's output units
model.load_state_dict(torch.load('../fly_CNN_model3_4_06Best.pth'))
model.eval()
model_CNN = model

# Load the PCA algorithm
pca_reloaded = load('../pca_model_added.sav')

# Load the SVM algorithm
modelUp_or_Down = load('../finalized_model_added.sav')

path_exp = '/Users/mariannecivitardevol/Documents/EPFL/Masters/projet semestre1/tracking_code/V3 - OB+check_flies+correction/F1/122/240223_x2p8_n1-60/F106_p0-7_60fps.mp4'
path_exp = '/Users/mariannecivitardevol/Documents/EPFL/Masters/projet semestre1/Project submission/data_metrics_and_videos/gui_leg/output5.mp4'

exp = Experiment(path_exp)
exp.process_video(pca_reloaded, modelUp_or_Down, model_CNN, cnn=True, svm=False, both=False)

print(exp.df)
