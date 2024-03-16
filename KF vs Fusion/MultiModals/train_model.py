import sys
import os
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F

from torchvision import datasets, transforms

from tqdm.notebook import tqdm

def train_image_model(model, train_loader, valid_loader, optimizer, criterion, epochs=5):
    for epoch in range(epochs):
        model.train()
        for images,  _, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            # print(f'image shape : {images.shape}')
            # print(f'label shape : {labels.shape}')
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, _, labels in valid_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        
        print(f"Epoch {epoch + 1}, Validation Accuracy: {accuracy * 100:.2f}%")

# Training loop for audio model
def train_audio_model(model, train_loader, valid_loader, optimizer, criterion, epochs=5):
    for epoch in range(epochs):
        model.train()
        for _,  audio, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            # print(f'audio shape : {audio.shape}')
            # print(f'label shape : {labels.shape}')
            optimizer.zero_grad()
            outputs = model(audio)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for _, audio, labels in valid_loader:
                outputs = model(audio)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        
        print(f"Epoch {epoch + 1}, Validation Accuracy: {accuracy * 100:.2f}%")
        
# Training loop for joint model
def train_joint_model(model, train_loader, valid_loader, optimizer, criterion, epochs=5):
    for epoch in range(epochs):
        model.train()
        for image,  audio, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            # print(f'image shape : {image.shape}')
            # print(f'audio shape : {audio.shape}')
            # print(f'label shape : {labels.shape}')
            optimizer.zero_grad()
            outputs = model((image, audio))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for image, audio, labels in valid_loader:
                outputs = model((image, audio))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        
        print(f"Epoch {epoch + 1}, Validation Accuracy: {accuracy * 100:.2f}%")
