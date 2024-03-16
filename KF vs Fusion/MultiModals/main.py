import sys
import os
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F

from model import LeNet5Image, LeNet5Audio, JointModel_early, JointModel_feature, JointModelFeatureAttention
from train_model import train_image_model, train_audio_model, train_joint_model
from config import *
sys.path.append(os.getcwd())
# from torchmetrics import Accuracy

if  __name__ == "__main__":
    
    # traindata[0] is image; traindata[1] is audio
    traindata = [np.load(data_dir+"/image/train_data.npy"), np.load(data_dir +
                                                                    "/audio/train_data.npy"), np.load(data_dir+"/train_labels.npy")]
    testdata = [np.load(data_dir+"/image/test_data.npy"), np.load(data_dir +
                                                                "/audio/test_data.npy"), np.load(data_dir+"/test_labels.npy")]

    # data form setting
    if flatten_audio:
        traindata[1] = traindata[1].reshape(60000, 112*112)
        testdata[1] = testdata[1].reshape(10000, 112*112)
    if normalize_image:
        traindata[0] /= 255.0
        testdata[0] /= 255.0
    if normalize_audio:
        traindata[1] = traindata[1]/255.0
        testdata[1] = testdata[1]/255.0
    if not flatten_image:
        traindata[0] = traindata[0].reshape(60000, 28, 28)
        testdata[0] = testdata[0].reshape(10000, 28, 28)
    if unsqueeze_channel:
        traindata[0] = np.expand_dims(traindata[0], 1)
        testdata[0] = np.expand_dims(testdata[0], 1)
        traindata[1] = np.expand_dims(traindata[1], 1)
        testdata[1] = np.expand_dims(testdata[1], 1)


    traindata[2] = traindata[2].astype(int)
    testdata[2] = testdata[2].astype(int)

    trainlist = [[traindata[j][i] for j in range(3)] for i in range(60000)]
    testlist = [[testdata[j][i] for j in range(3)] for i in range(10000)]

    #######################

    #### TODO  : afficher quelques images et les annotations
    # Data loading and visualization
    def show_images(images, labels, num_images=1, image_shape=(112, 112)):
        fig, axes = plt.subplots(1, num_images)
        for i in range(num_images):
            # Reshape the flattened image to the original shape
            image = images[i].reshape(image_shape)
            axes[i].imshow(image, cmap='gray')
            axes[i].set_title(f"Label: {labels[i]}")
        plt.show()
        
    # sample_images = traindata[1][:2]
    # sample_labels = traindata[2][:2]
    # show_images(sample_images, sample_labels)

    #######################

    print("Initialize Dataloader ...", end='')
    validdata = DataLoader(trainlist[55000:60000], shuffle=False, batch_size=batch_size)
    testdata = DataLoader(testlist, shuffle=False, batch_size=batch_size)
    traindata = DataLoader(trainlist[0:55000], shuffle=train_shuffle, batch_size=batch_size)
    print("100%")

    print("Initialize Model ...", end='')
    image_model = LeNet5Image()
    audio_model = LeNet5Audio()
    joint_model_early = JointModel_early()
    joint_model_feature = JointModel_feature()
    joint_model_feature_attention = JointModelFeatureAttention()
    print("100%")
    print(f"Number of parameters in image model: {sum(p.numel() for p in image_model.parameters())}")
    print(f"Number of parameters in audio model: {sum(p.numel() for p in audio_model.parameters())}")
    print(f"Number of parameters in the Early Joint Model: {sum(p.numel() for p in joint_model_early.parameters())}")
    print(f"Number of parameters in the Feature Joint Model: {sum(p.numel() for p in joint_model_feature.parameters())}")
    print(f"Number of parameters in the Feature Attention Joint Model: {sum(p.numel() for p in joint_model_feature_attention.parameters())}")
   
    # Train and evaluate image model
    print("Initialize Optimizer ...", end='')
    optimizer_image = torch.optim.Adam(image_model.parameters(), lr=0.001)
    optimizer_audio = torch.optim.Adam(audio_model.parameters(), lr=0.001)
    optimizer_joint_early = torch.optim.Adam(joint_model_early.parameters(), lr=0.001)
    optimizer_joint_feature = torch.optim.Adam(joint_model_feature.parameters(), lr=0.001)
    optimizer_joint_feature_attention = torch.optim.Adam(joint_model_feature_attention.parameters(), lr=0.001)
    print("100%")

    print("Initialize Loss ...", end='')
    criterion = nn.CrossEntropyLoss()
    print("100%")
    
    print("Train Model : [START]")
    
    # train_image_model(image_model, traindata, validdata, optimizer_image, criterion, epochs=5)
    # image_model.eval()
    # with torch.no_grad():
    #     correct = 0
    #     total = 0
    #     for images, audio, labels in testdata:
    #         outputs = image_model(images.float())
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    #     accuracy = correct / total
    #     print(f"Test Accuracy for Image Model: {accuracy * 100:.2f}%")
        
    # train_audio_model(audio_model, traindata, validdata, optimizer_audio, criterion, epochs=5)
    # audio_model.eval()
    # with torch.no_grad():
    #     correct = 0
    #     total = 0
    #     for images, audio, labels in testdata:
    #         outputs = audio_model(audio.float())
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    #     accuracy = correct / total
    #     print(f"Test Accuracy for Audio Model: {accuracy * 100:.2f}%")
        
    # train_joint_model(joint_model_early, traindata, validdata, optimizer_joint_early, criterion, epochs=5)
    # joint_model_early.eval()
    # with torch.no_grad():
    #     correct = 0
    #     total = 0
    #     for images, audio, labels in testdata:
    #         outputs = joint_model_early((images.float(), audio.float()))
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    #     accuracy = correct / total
    #     print(f"Test Accuracy for early fusion Model: {accuracy * 100:.2f}%")
        
    train_joint_model(joint_model_feature, traindata, validdata, optimizer_joint_feature, criterion, epochs=5)
    joint_model_feature.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, audio, labels in testdata:
            outputs = joint_model_feature((images.float(), audio.float()))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total
        print(f"Test Accuracy for feature fusion Model: {accuracy * 100:.2f}%")
    
    print("Training process is completed!")  
    
    # train_joint_model(joint_model_feature_attention, traindata, validdata, optimizer_joint_feature_attention, criterion, epochs=5)
    # joint_model_feature.eval()
    # with torch.no_grad():
    #     correct = 0
    #     total = 0
    #     for images, audio, labels in testdata:
    #         outputs = joint_model_feature_attention((images.float(), audio.float()))
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    #     accuracy = correct / total
    #     print(f"Test Accuracy for feature attention fusion Model: {accuracy * 100:.2f}%")
    
    print("Training process is completed!")    
