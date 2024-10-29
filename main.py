# main.py
from data_loader import train_loader, val_loader
from new_model import CNN
from trainyy import train_model
import torch.nn as nn
import torch.optim as optim
import config

model = CNN(config.num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=config.num_epochs, checkpoint_dir='output/checkpoints/')
