# Drosophila Detection and Tracking System

A real-time and video-based tracking system for *Drosophila* (fruit flies) in behavioral experiments. This project enables detection, tracking, and orientation analysis of multiple flies using computer vision and machine learning.

## Project Objectives

- **Real-time tracking** of multiple flies (typically 2) from live camera feed
- **Video-based tracking** for post-processing of recorded experiments
- **Orientation detection** - determine fly head position and facing direction (stomach vs back visible)
- **Magnet/robot tracking** - track a magnet or robot alongside flies for controlled experiments
- **ML classification** - use CNN (ResNet50) and SVM models to classify fly facing direction
- **Robot position planning** - calculate feasible robot positions relative to the fly for experiment automation

## Code Structure

```
detection_drosophila/
│
├── live_tracking/                    # Real-time tracking from camera feed
│   ├── pipeline_final.py             # Core tracking pipeline with Experiment class
│   └── processing_final.py           # ML model loading and GUI interface functions
│
├── video_tracking/                   # Post-processing of recorded videos
│   ├── pipeline_video_2flies.py      # Pipeline for tracking 2 flies (no robot)
│   ├── pipeline_video_magnet.py      # Pipeline for tracking 1 fly + 1 magnet/robot
│   ├── track_video_2flies.py         # Script to run 2-fly video processing
│   └── track_video_magnet.py         # Script to run magnet video processing
│
├── classifiers_creation/             # ML model training and data collection
│   ├── CNN/
│   │   └── CNN_classifier.ipynb      # ResNet50-based CNN classifier training
│   ├── SVM/
│   │   └── SVM_classifier.ipynb      # SVM classifier with PCA training
│   └── get_data_functions/           # Training data extraction utilities
│       ├── dataAcq_func_2flies.py    # Helper functions for 2-fly data extraction
│       ├── dataAcq_func_magnet.py    # Helper functions for magnet data extraction
│       ├── get_data_2flies.py        # Script to extract fly images from 2-fly videos
│       └── get_data_magnet.py        # Script to extract fly images from magnet videos
│
└── run.sh                            # Shell script for live tracking setup
```

### Module Details

| Module | Description |
|--------|-------------|
| **live_tracking** | Processes frames from camera feed (stdin), performs real-time detection and tracking, outputs visualization and tracking data |
| **video_tracking** | Processes entire recorded videos, saves annotated videos and trajectory DataFrames for analysis |
| **classifiers_creation** | Tools for extracting training images from videos and training CNN/SVM classifiers |

## Installation

### Dependencies

Install the required Python packages:

```bash
pip install numpy opencv-python pandas scipy torch torchvision scikit-learn joblib pillow imageio
```

### Required Libraries

| Library | Purpose |
|---------|---------|
| `opencv-python` | Image processing, contour detection, ellipse fitting |
| `numpy` | Array operations |
| `pandas` | Trajectory data storage and manipulation |
| `scipy` | Hungarian algorithm for ID assignment, distance calculations |
| `torch`, `torchvision` | PyTorch for CNN model (ResNet50) |
| `scikit-learn` | SVM classifier, PCA, StandardScaler |
| `joblib` | Model serialization |
| `pillow` | Image handling |
| `imageio` | Video writing |

## Usage

### 1. Live Tracking

For real-time tracking from a FLIR camera:

```bash
./run.sh
```

This sets up the camera capture and pipes frames to the Python GUI:

```bash
./flir-control 20 | python gui.py
```

### 2. Video Processing

#### Track 2 Flies (no robot)

```python
from video_tracking.pipeline_video_2flies import *

# Initialize experiment with video path
exp = Experiment("path/to/video.mp4")

# Process video with ML models
exp.process_video(
    pca_model,           # PCA model for SVM
    svm_model,           # SVM classifier
    cnn_model,           # CNN classifier
    cnn=True,            # Use CNN predictions
    svm=False,           # Don't use SVM predictions
    both=False           # Don't use ensemble
)

# Outputs: tracked video + trajectory DataFrame
```

#### Track 1 Fly + Magnet/Robot

```python
from video_tracking.pipeline_video_magnet import *

exp = Experiment("path/to/video.mp4")
exp.process_video(pca_model, svm_model, cnn_model, cnn=True, svm=False, both=False)
```

### 3. Training New Classifiers

#### Extract Training Data

```python
# For 2-fly videos
python classifiers_creation/get_data_functions/get_data_2flies.py

# For magnet videos
python classifiers_creation/get_data_functions/get_data_magnet.py
```

#### Train Models

Open and run the Jupyter notebooks:
- `classifiers_creation/CNN/CNN_classifier.ipynb` - Train ResNet50 CNN
- `classifiers_creation/SVM/SVM_classifier.ipynb` - Train SVM with PCA

## How It Works

### Tracking Pipeline

1. **Preprocessing** - Hide walls, apply threshold, detect contours
2. **Detection** - Find the 2 largest contours (fly + robot/magnet), apply erosion/dilation corrections if needed
3. **ID Assignment** - Hungarian algorithm maintains identity across frames
4. **Orientation Analysis** - Fit ellipse to fly contour, determine head position via intensity analysis
5. **ML Classification** - Every 10 frames, classify facing direction (stomach=1, back=2)
6. **Robot Planning** - Calculate feasible angles/positions for magnet movement
7. **Output** - Annotated frames with trajectories, angles, and predictions

### ML Models

| Model | Architecture | Purpose |
|-------|--------------|---------|
| **CNN** | ResNet50 (fine-tuned) | Binary classification: stomach vs back visible |
| **SVM** | SVM with PCA features | Alternative classifier for facing direction |

Both models can be used individually or as an ensemble for improved accuracy.

## Output

- **Annotated video** - Visual tracking with trajectories and orientation markers
- **Trajectory DataFrame** - CSV/DataFrame containing frame-by-frame positions, orientations, and predictions
- **Performance metrics** - Tracking accuracy and classification metrics
