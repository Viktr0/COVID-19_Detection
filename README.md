# COVID-19_Detection

Thanks to technological improvements and the published quality databases in recent years, deep learning could provide a solution to many problems. New algorithms in the field of computer vision have increased the accuracy of classification, segmentation, and object detection to such an extent that it surpasses the human-level of accuracy in many cases. As in many other tasks, deep learning tools have become significant components of medical imaging.

My research aimed to develop an algorithm that can be applied to recognize COVID-19. The virus is filtered out by a neural network that uses a database of lung X-ray images.

During the semester, I searched for famous convolutional neural networks and databases that are suitable for the task. I implemented my model for examining the selected databases using other architecture.

In my dissertation, I describe the deep learning tools used in image processing. I turn to convolutional neural networks and the layers that build them up. I analyze and compare the architectures. I present the model I have implemented for examining images. Finally, I draw the conclusion of the results obtained.

The research consists of three main parts:
* Preprocessing and splitting the dataset
* Classification and evaluation
* Visual explanations

Each of the subtasks are implemented in different Python notebooks.

## Data Preprocessing
Two datasets were used in this research:
* less accurate, but cheap X-ray images
* more precise and more expensive CT scans

It is important to separate the database into distinct sets, these are the train, validation and test datasets. 
Label encoding and randomization, normalization or standardization of the data are also commonly used procedures.

<p align="center">
  <img width=60% src="https://github.com/Viktr0/COVID-19_Detection/assets/47856193/3bc43359-36dc-4150-80f6-2595624b1d2a"/>
</p>

The data set consisting of radiographs is not uniform enough. The dataset compiled from different sources has different characteristics, as well as objects and information that are not relevant to the task.

One of the consequences of this may be that the neural network will not classify based on changes caused by covid.

<p align="center">
  <img width=60% src="https://github.com/Viktr0/COVID-19_Detection/assets/47856193/ead92f06-2f6f-4ec9-b310-13c034b3244f"/>
</p>

That's why I looked for a reliable, doctor-verified data set that contains CT scans.

## Model
VGG is an easy-to-implement, reliable, sequential deep convolutional neural network architecture.
So I implemented the VGG11 model using the nn module of PyTorch.
<p align="center">
  <img width=60% src="https://github.com/Viktr0/COVID-19_Detection/assets/47856193/4d43d387-55cb-46d7-9249-85e4054289ec"/>
</p>

The first 29 layers of the model are responsible for the feature extraction and 7 more layers for the classification task.

The loss curve is a critical tool for monitoring the training process and diagnosing issues such as convergence, overfitting, and underfitting in machine learning models.

Loss             |  Accuracy
:-------------------------:|:-------------------------:
![image](https://github.com/Viktr0/COVID-19_Detection/assets/47856193/56e551e6-8dbe-4069-a171-175df56e79ec)  |  ![image](https://github.com/Viktr0/COVID-19_Detection/assets/47856193/24f258a2-f789-48eb-8981-fcef8c9d476d)

## Evaluation
A confusion matrix is a table that is often used to describe the performance of a classification model on a set of test data for which the true values are known. It allows visualization of the performance of an algorithm.

* True Positive (TP): Predicted positive and actually positive.
* False Positive (FP): Predicted positive but actually negative.
* False Negative (FN): Predicted negative but actually positive.
* True Negative (TN): Predicted negative and actually negative.

X-ray | CT 
--- | --- 
![image](https://github.com/Viktr0/COVID-19_Detection/assets/47856193/0e6e8fd5-9c6b-46f0-833f-226ef4593f7a) | ![image](https://github.com/Viktr0/COVID-19_Detection/assets/47856193/090837c5-f6a2-4ec7-966c-a7edf8af610a)

The values of the confusion matrix are the results of the evaluation of the model on the test set, whis was never seen by the model before.
From the confusion matrix, you can derive various evaluation metrics.  

### Accuracy
The proportion of the total number of predictions that were correct.

Accuracy = (TP + TN) / (TP + FP + TN + FN)

**Dataset** | **Acc** 
--- | ---
X-ray | 98.47%
CT | 82.67%

### Precision
The proportion of positive cases that were correctly identified.

Precision = TP / (TP + FP)

**Dataset** | **Class** | **Prec**
--- | --- | ---
X-ray | normal | 1
X-ray | COVID | 0.96
CT | noraml | 0.86
CT | COVID | 0.79

### Recall
Also called as sensitivity. The proportion of actual positive cases that were correctly identified.

Recall = TP / (TP + FN)

**Dataset** | **Class** | **Rec**
--- | --- | ---
X-ray | normal | 0.96
X-ray | COVID | 1
CT | noraml | 0.86
CT | COVID | 0.8

## PCA

 PCA is a powerful technique for simplifying and analyzing complex datasets by capturing and representing their essential characteristics in a lower-dimensional space.

<p align="center">
  <img width=70% src="https://github.com/Viktr0/COVID-19_Detection/assets/47856193/d36045f4-c271-41d8-bcd1-5bc40fec81c3"/>
</p>

On the left side, in the PCA diagram run on the X-ray database, the groups are more visibly separated from each other, which also reflects the results of the classifications.

## Grad CAM


Grad-CAM (Gradient-weighted Class Activation Mapping) is a technique used in convolutional neural networks (CNNs) to visualize and understand the regions of an input image that are important for predicting a particular class. It helps in understanding where the model is focusing its attention when making predictions.

Input | Layer 4 | Layer 11 | Layer 18 | Layer 25 
--- | --- | --- | --- |--- 
![image](https://github.com/Viktr0/COVID-19_Detection/assets/47856193/db5bd88e-e6d2-4e54-bc09-ec47598ef611) | ![image](https://github.com/Viktr0/COVID-19_Detection/assets/47856193/adb7f7da-6e05-4ded-8c5e-f7a2f82f0698) | ![image](https://github.com/Viktr0/COVID-19_Detection/assets/47856193/0d662082-2dae-481f-b51c-8d8986077c9f) | ![image](https://github.com/Viktr0/COVID-19_Detection/assets/47856193/db973cc4-c377-4ebd-94f4-af6cde2b13a8) | ![image](https://github.com/Viktr0/COVID-19_Detection/assets/47856193/42522a66-4362-4a1b-9453-55359430c002)


Input | Layer 4 | Layer 11 | Layer 18 | Layer 25 
--- | --- | --- | --- |--- 
![image](https://github.com/Viktr0/COVID-19_Detection/assets/47856193/2c4845a1-f431-4d67-bc25-fa0e52b5645f) | ![image](https://github.com/Viktr0/COVID-19_Detection/assets/47856193/9af32125-00a7-427c-a993-e7642eb33c5d) | ![image](https://github.com/Viktr0/COVID-19_Detection/assets/47856193/7703a3b3-47f0-41c4-bc98-1e9c12733b68) | ![image](https://github.com/Viktr0/COVID-19_Detection/assets/47856193/1788fd1a-2964-4cfb-ad47-68cf2624192a) | ![image](https://github.com/Viktr0/COVID-19_Detection/assets/47856193/60daf211-c36c-4b04-b0a4-faa8688b5200)

