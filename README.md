# Prediction-of-diseases
1. The code is present on Disease_Prediction.ipynb
2. You may download the code on your local machine and run on platforms that support ipynb files.
3. Dataset used: https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia from kaggle

---

#  Disease Prediction using ResNet50

##  Overview
This project implements a **deep learning-based disease prediction model** using the **ResNet50** architecture.  
It classifies medical images into different disease categories, supporting diagnosis and early detection efforts.

The model is built using **TensorFlow/Keras**, with preprocessing, augmentation, training, and evaluation steps.

---
## Purpose of the Project
- The primary motivation for this project was to leverage deep learning to assist in the early detection of diseases from medical images.
- Early and accurate diagnosis can significantly improve patient outcomes, reduce the cost of treatment, and help medical professionals make faster, data-driven decisions.
This project serves as both a technical proof-of-concept and a practical tool that could be integrated into clinical workflows or used for academic research and experimentation.
---
## Dataset
- **Type**: Medical image dataset containing labeled images for various diseases.
- **Structure**:
```
  .
├── Disease_Prediction.ipynb
├── README.md
├── requirements.txt        # optional
└── dataset/                # not tracked in git
    ├── train/
    ├── valid/
    └── test/
```
---

- **Data Augmentation**: Implemented using `ImageDataGenerator` to improve model generalization.

---

## Requirements
Install the dependencies before running the notebook:
```
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn
```
---
## Model Architecture

Base Model: ResNet50 (pre-trained on ImageNet)

Layers Added:
- Flatten
- Dense (Fully Connected Layer)
- Dropout (for regularization)
- Output Layer (softmax activation for multi-class classification)
- Loss Function: categorical_crossentropy
- Optimizer: Adam
- Metrics: Accuracy

---

## Training Process

Data Preprocessing:
- Resize images to ResNet50 input shape.
- Normalize and preprocess with preprocess_input.
- Apply augmentation (rotation, zoom, flip, etc.).

## Model Training:

- Use EarlyStopping to prevent overfitting.
- Employ validation split to monitor progress.

## Evaluation:

- Generate classification report (precision, recall, F1-score).
- Plot accuracy/loss curves.

## Results

- Evaluation Metrics: Accuracy, Precision, Recall, F1-score.
- Visualization: Training and validation accuracy/loss plots.

## How to Run

Clone the repository:
```
git clone https://github.com/yourusername/Disease_Prediction.git
cd Disease_Prediction
```
## Install dependencies:
```
pip install -r requirements.txt
```
## Run the notebook:
 
 jupyter notebook Disease_Prediction.ipynb
 Follow the steps in the notebook to train and evaluate the model.

## Notes
- Update the dataset path in the notebook to match your local/Google Drive structure.
- Use GPU acceleration for significantly faster training.
