
# Advanced Deep Learning with Computer Vision to Spot Nuclei


![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)



## Summary
This project implement an advanced deep learning with computer vision architecture model (i.e. UNet) to advance medical discovery in Python.
## Abstract
An algorithm that can automatically detect nuclei to expedite research on a wide 
range of diseases, including cancer, heart disease, and rare disorders. Such tool has 
the potential to significantly improve the speed at which cures are developed, 
benefiting those suffering from various health conditions, including chronic 
obstructive pulmonary disease, Alzheimer's, diabetes, and even the common cold.

Hence, the identification of cell nuclei is a crucial first step in many research studies 
because it enables researchers to analyze the DNA contained within the nucleus, 
which holds the genetic information that determines the function of each cell. By 
identifying the nuclei of cells, researchers can examine how cells respond to 
different treatments and gain insights into the underlying biological processes at 
play. An automated AI model for identifying nuclei, offers the potential to streamline 
drug testing and reduce the time for new drugs to become available to the public.

Thus, this project create a model for semantic segmentation for images containing cell neuclei.

## Data Set
Datasets used for developing this model can be obtain from [2018 Data Science Bowl](https://www.kaggle.com/competitions/data-science-bowl-2018/overview)

## Run Locally

+ Clone the project

```bash
  git clone https://github.com/liewwy19/Advanced-Deep-Learning-with-Computer-Vision-to-Spot-Nuclei.git
```
+ Create an environment and install the needed packages and its dependencies

```bash
  pip install -r requirements.txt 
```
+ Install the tensorflow-examples packages

```bash
  pip install git+https://github.com/tensorflow/examples.git
```
+ Download and extract the dataset into a sub-folder named "datasets" within the project's folder. Your project folder structure should look like the following:
```bash
Project Folder
  |
  |--- datasets
  |      |
  |      |--- data-science-bowl-2018-2
  |             |
  |             |--- test 
  |             |      |
  |             |      |--- inputs
  |             |      |--- masks   
  |             |
  |             |--- train     
  |                    |
  |                    |--- inputs
  |                    |--- masks
  |
  |--- saved_models     
  
```




## Methodology
+ Import packages
    + set constant variables
    + define classes and functions
+ Data Loading
    + load train dataset
    + load test dataset
+ Data Preprocessing
    + convert dataset to tensor dataset
    + build the datasets
+ Model Development
    + instantiate the base Model
    + construct the entire U-net using functional API
+ Model Compilation
+ Model Training
+ Model Evaluation
+ Model Deployment
    + perform prediction with test data
    + model saving


## Model

### The U-Net Architecture
![](https://media.geeksforgeeks.org/wp-content/uploads/20220614121231/Group14.jpg)

(Image source: https://arxiv.org/pdf/1505.04597.pdf)

![](https://github.com/liewwy19/Advanced-Deep-Learning-with-Computer-Vision-to-Spot-Nuclei/blob/main/model.png?raw=True)

## Process

### Prediction - "Pre-train"
![](https://github.com/liewwy19/Advanced-Deep-Learning-with-Computer-Vision-to-Spot-Nuclei/blob/main/assets/prediction-0-pre-train.png?raw=True)

### Prediction - After 1 epoch
![](https://github.com/liewwy19/Advanced-Deep-Learning-with-Computer-Vision-to-Spot-Nuclei/blob/main/assets/prediction-1-after_1_epoch.png?raw=True)

### Prediction - Trained model
![](https://github.com/liewwy19/Advanced-Deep-Learning-with-Computer-Vision-to-Spot-Nuclei/blob/main/assets/prediction-2-trained.png?raw=True)
## Analysis
The model able to achieve over 90% accuracy. Matter a fact, for certain input images, the predicted masks from the model are even better than the true mask. (Refer to the Results section below.) Specifically in "Prediction 2", area near the lower right section of the predicted mask.

![](https://github.com/liewwy19/Advanced-Deep-Learning-with-Computer-Vision-to-Spot-Nuclei/blob/main/chart_model_evaluation.png?raw=True)
## The Results

#### Prediction 1
![](https://github.com/liewwy19/Advanced-Deep-Learning-with-Computer-Vision-to-Spot-Nuclei/blob/main/assets/prediction-3.png?raw=True)

#### Prediction 2
![](https://github.com/liewwy19/Advanced-Deep-Learning-with-Computer-Vision-to-Spot-Nuclei/blob/main/assets/prediction-4.png?raw=True)

#### Prediction 3
![](https://github.com/liewwy19/Advanced-Deep-Learning-with-Computer-Vision-to-Spot-Nuclei/blob/main/assets/prediction-5.png?raw=True)


## Contributing

This project welcomes contributions and suggestions. 

    1. Open issues to discuss proposed changes 
    2. Fork the repo and test local changes
    3. Create pull request against staging branch


## Acknowledgements
 - [2018 Data Science Bowl](https://www.kaggle.com/competitions/data-science-bowl-2018/overview)
 - [U-Net Architecture For Image Segmentation](https://blog.paperspace.com/unet-architecture-image-segmentation/#:~:text=Computer%20vision%20is%20one%20such,datasets%20to%20perform%20complex%20tasks.)
 - [Image Segmentation Using TensorFlow](https://www.geeksforgeeks.org/image-segmentation-using-tensorflow/)
 - [Selangor Human Resource Development Centre (SHRDC)](https://www.shrdc.org.my/)

