import torch
import os
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np
import seaborn as sns
from tqdm import tqdm
import config
import data_loader


os.makedirs('output/checkpoints/', exist_ok=True)
os.makedirs('output/', exist_ok=True)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, final_model_name='best_model.pth', checkpoint_dir='output/checkpoints/'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_acc = 0.0
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    all_predictions = []
    all_labels = []
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # Find the latest checkpoint file
    checkpoint_files = [file for file in os.listdir(checkpoint_dir) if file.startswith('checkpoint_epoch_')]
    if checkpoint_files:
        latest_checkpoint_file = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint_file)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_val_acc = checkpoint['best_val_acc']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        train_accuracies = checkpoint['train_accuracies']
        val_accuracies = checkpoint['val_accuracies']
        print(f"Checkpoint loaded. Resuming training from epoch {epoch + 1}")
    else:
        epoch = 0

    for epoch in range(epoch, num_epochs):
        model.train()
        running_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch", position=0, leave=True, colour='green') as pbar:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                pbar.update(1) 
                pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Batch Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_acc = correct / total
        train_accuracies.append(train_acc)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        val_acc = correct / total
        val_accuracies.append(val_acc)
        print(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}")
        # print(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}")
        print(f"Total batches processed in validation: {batch_count}")

        # Adjust learning rate based on validation loss
        scheduler.step(val_loss)

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'train_losses': train_losses,
                'train_accuries': train_accuracies,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies
            }, checkpoint_path)

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join('output', final_model_name))
            print(f"Model saved with accuracy: {best_val_acc:.4f}")

    # Save the final model in the output folder
    final_model_path = os.path.join('output', final_model_name.split('.')[0] + '_final.pth')
    torch.save(model.state_dict(), final_model_path)

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')
    plt.savefig(os.path.join('output', 'loss_plot.png'))

    # Plot validation accuracy
    plt.figure(figsize=(10, 5))
    plt.figure(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.savefig(os.path.join('output', 'accuracy_plot.png'))

    plt.show()

    print(f"All labels: {all_labels}")
    print(f"All predictions: {all_predictions}")
    print(f"Unique labels: {np.unique(all_labels)}")
    print(f"Unique predictions: {np.unique(all_predictions)}")


      # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    if cm.size ==0:
        print("Error: Confusion matrix is empty. Check your data and model.")
    else: 
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='g', xticklabels=config.classes, yticklabels=config.classes, cmap='Blues')
        plt.xlabel('Prediction', fontsize=13)
        plt.ylabel('Actual', fontsize=13)
        plt.title(f'Confusion Matrix - Epoch {epoch+1}', fontsize=17)
        plt.savefig(os.path.join('output', 'confusion_matrix.png'))
        plt.show()



    # Convert labels to one-hot encoding
    one_hot_labels = label_binarize(all_labels, classes=np.arange(len(config.classes)))

    # Convert predictions to probabilities
    softmax = nn.Softmax(dim=0)

    #  Convert class labels to probabilities
    probabilities = torch.zeros(len(all_predictions), len(config.classes))
    probabilities.scatter_(1, torch.unsqueeze(torch.tensor(all_predictions), 1), 1)
    probabilities = softmax(probabilities).detach().numpy()


    # Plot ROC curve for each class
    plt.figure(figsize=(10, 8))
    for i in range(len(config.classes)):
        fpr, tpr, _ = roc_curve(one_hot_labels[:, i], probabilities[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label='ROC curve for class %d (area = %0.2f)' % (i, roc_auc))

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - Multiclass')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join('output', 'auroc_curve.png'))
    plt.show()
