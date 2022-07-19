
# Brain Tumor Classification with Convolutional Neural Network

In this project, I want to use Neural Network to be able to identify and classify different types of neural network. The end goal is I want to be able to create an app where a user can easily use their phone camera and scan a picture of a brain MRI Scans and immediately detect and classify the tumor. 
This is the roadmap of the project

## The Roadmap

![Logo](https://github.com/vincentpatrick/BrainTumourClassificationProject/blob/main/brain%20tumor%20detection%20process.png)

| Model Building | TensorFlow, CNN, Data Augmentation, TF dataset |
|----------------|------------------------------------------------|
| Backend Server | FastAPI, GCP                                   |
| Front End      | React                                          |

Steps:
1.	Data collection
For data collection, I retrieve the brain tumor datasets from Kaggle
2.	Data preprocessing
I split the data into segments for training, validation and testing.
3.	Model Building
Start building convolutional neural network
4.	FastAPI and GCP
Use fastAPi to Integrate the CNN model to the localhost, enabling users to upload an image into the CNN and receive its prediction immediately




## Documentation

### First Classification CNN model Test run

Deploy the model into a google cloud platform to enable AI model to be usable in mobile app.

1st model run performance.
![Logo](https://github.com/vincentpatrick/BrainTumourClassificationProject/blob/main/Screenshots/performance.PNG)

Based on the plot, training accuracy is 1 and training loss is 0 after 20 epochs. 
Therefore, we only need to train the model for 20 epochs.

The model is Run and it is able to predict the images of different tumor types correctly.

![Logo](https://github.com/vincentpatrick/BrainTumourClassificationProject/blob/main/Screenshots/Capture.PNG)

### FastAPI Integration

I code with python to integrate my CNN model to a localhost using FastAPI,
enabling users to upload an image into the CNN and receive its prediction immediately.
![Logo](https://github.com/vincentpatrick/BrainTumourClassificationProject/blob/main/Screenshots/fastapi1.png)

The user can pick a type of Tumor image to send into the CNN model.
![Logo](https://github.com/vincentpatrick/BrainTumourClassificationProject/blob/main/Screenshots/fastapi2.png)

The model directly sends its tumor type prediction of that image to the user.
![Logo](https://github.com/vincentpatrick/BrainTumourClassificationProject/blob/main/Screenshots/fastapi3.PNG)

The model is performed on a test dataset, it was able to predict most of the brain tumour correctly based on the type of tumor.

### Image Data Generator FastAPI
In order to improve the Convolutional Neural Network, I use an Image data generator API for data augmentation enabling the model to have more data to train and test with.
The Process is the same as before, only Image Data Generator is used to provide the model with more data for training, test and validation.

The model is build, compiled and trained. The accuracy and loss plot are as follow
![Logo](https://github.com/vincentpatrick/BrainTumourClassificationProject/blob/main/Screenshots/accuracy_and_loss.PNG)

The new model is run and is now able to predict more images of tumor.
![Logo](https://github.com/vincentpatrick/BrainTumourClassificationProject/blob/main/Screenshots/new_prediction.PNG)

