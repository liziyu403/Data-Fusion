import sys
import os
import numpy as np
import torch.nn as nn

import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F

# Neural networks based on images as input
class LeNet5Image(nn.Module):
    def __init__(self):
        super(LeNet5Image, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # x = (batch_size, channels, height, width)
        x = F.relu(self.conv1(x.float())) 
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# Neural networks based on audio as input
class LeNet5Audio(nn.Module):
    def __init__(self):
        super(LeNet5Audio, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(10816, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # print('1',x.shape)
        x = F.relu(self.conv1(x.float()))
        # print('2',x.shape)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # print('3', x.shape)
        x = F.relu(self.conv2(x))
        # print('4', x.shape)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # print('5',x.shape)
        x = x.view(x.size(0), -1)
        # print('6',x.shape)
        x = F.relu(self.fc1(x))
        # print('7',x.shape)
        x = F.relu(self.fc2(x))
        # print('8',x.shape)
        x = self.fc3(x)
        return x
    
# Neural networks based on images and audio as input: use early fusion
class JointModel_early(nn.Module):
    def __init__(self):
        super(JointModel_early, self).__init__()

        # Common layers after merging
        self.conv1 = nn.Conv2d(17, 32, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=5, stride=1)
        
        self.fc1 = nn.Linear(768, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, input_data):
        img, audio = input_data

        # Image branch
        x_img = img.float()

        # Audio branch
        audio_concatenated = torch.zeros(audio.size(0), 16, 28, 28) # Split audio into 16 28x28 chunks

        for j in range(audio.size(0)):
            for i in range(16):
                row_start = (i // 4) * 28
                col_start = (i % 4) * 28
                audio_concatenated[j, i, :, :] = audio[j, 0, row_start:row_start+28, col_start:col_start+28]
        # perm_indices = torch.tensor([ 9,  0,  4,  5, 13, 11,  1,  8,  2, 15,  7, 14,  3, 10, 12,  6])
        # perm_indices = torch.tensor([12,  7,  4, 10,  0, 15, 13, 11,  2, 14,  3,  5,  8,  9,  6,  1])
        perm_indices = torch.tensor([ 7,  4, 12,  5, 11,  1, 13,  9, 15,  2,  6,  3,  0, 10,  8, 14])
        shuffled_audio = audio_concatenated[:, perm_indices, :, :]
        
        x = torch.cat((x_img, shuffled_audio), dim=1)
                
        # Common layers for fusion
        x = F.relu(self.conv1(x.float()))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x.float()))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    
class JointModel_feature(nn.Module):
    def __init__(self):
        super(JointModel_feature, self).__init__()

        # Image branch
        self.conv1_img = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.conv2_img = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        
        # Audio branch
        self.conv1_audio = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.conv2_audio = nn.Conv2d(6, 16, kernel_size=5, stride=1)

        # Common layers after merging
        self.fc1 = nn.Linear(11216, 10)
    

    def forward(self, input_data):
        img, audio = input_data

        # Image branch
        x_img = F.relu(self.conv1_img(img.float()))
        x_img = F.max_pool2d(x_img, kernel_size=2, stride=2)
        x_img = F.relu(self.conv2_img(x_img))
        x_img = F.max_pool2d(x_img, kernel_size=2, stride=2)
        x_img = x_img.view(x_img.size(0), -1)

        # Audio branch
        x_audio = F.relu(self.conv1_audio(audio.float()))
        x_audio = F.max_pool2d(x_audio, kernel_size=2, stride=2)
        x_audio = F.relu(self.conv2_audio(x_audio))
        x_audio = F.max_pool2d(x_audio, kernel_size=2, stride=2)
        x_audio = x_audio.view(x_audio.size(0), -1)

        # Concatenate both branches
        x = torch.cat((x_img, x_audio), dim=1)

        # Common layers
        x = self.fc1(x)
     
        return x
    
    
class JointModelFeatureAttention(nn.Module):
    def __init__(self):
        super(JointModelFeatureAttention, self).__init__()

        # Image branch
        self.conv_img1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv_img2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)

        # Audio branch
        self.conv_audio1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv_audio2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)

        # Attention module
        self.attention = nn.MultiheadAttention(embed_dim=16, num_heads=4)

        # Common layers after merging
        self.fc1 = nn.Linear(784, 10)

    def forward(self, input_data):
        img, audio = input_data

        # Image branch
        x_img = F.relu(self.conv_img1(img.float()))
        x_img = F.max_pool2d(x_img, kernel_size=2, stride=2)
        x_img = F.relu(self.conv_img2(x_img.float()))
        x_img = F.max_pool2d(x_img, kernel_size=2, stride=2)
        x_img = x_img.view(x_img.size(0), 16, -1)

        # Audio branch
        x_audio = F.max_pool2d(audio.float(), kernel_size=4, stride=4)
        x_audio = F.relu(self.conv_audio1(x_audio.float()))
        x_audio = F.max_pool2d(x_audio, kernel_size=2, stride=2)
        x_audio = F.relu(self.conv_audio2(x_audio.float()))
        x_audio = F.max_pool2d(x_audio, kernel_size=2, stride=2)
        x_audio = x_audio.view(x_img.size())

        x_img = x_img.permute(0, 2, 1)
        x_audio = x_audio.permute(0, 2, 1)
        
        # Attention module
        x_attended, _ = self.attention(x_img, x_audio, x_audio)
        
        # Flatten attended features
        x_attended = x_attended.view(x_attended.size(0), -1)

        # Common layers
        x_combined = self.fc1(x_attended)

        return x_combined
