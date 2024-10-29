import os
import torch
import torchaudio.transforms as T
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
import random
from tqdm import tqdm

class SpectrogramDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.file_paths = []
        self.labels = []

        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            if os.path.isdir(cls_dir):
                files = os.listdir(cls_dir)
                random.shuffle(files)  # Shuffle files within the class
                for file_name in files:
                    file_path = os.path.join(cls_dir, file_name)
                    self.file_paths.append(file_path)
                    self.labels.append(self.class_to_idx[cls_name])

        # Check and filter out corrupted images
        self.file_paths, self.labels = self.check_images(self.file_paths, self.labels)

        combined = list(zip(self.file_paths, self.labels))
        random.shuffle(combined)  # Shuffle combined list of file paths and labels
        self.file_paths, self.labels = zip(*combined)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(file_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except (IOError, UnidentifiedImageError) as e:
            print(f"Skipping corrupted image at index {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))  # Attempt to get the next item

    def check_images(self, file_paths, labels):
        valid_file_paths = []
        valid_labels = []
        for path, label in tqdm(zip(file_paths, labels), desc="Checking images"):
            try:
                with Image.open(path) as img:
                    img.verify()
                valid_file_paths.append(path)
                valid_labels.append(label)
            except (IOError, UnidentifiedImageError) as e:
                print(f"Corrupted image detected and skipped: {path}")
        return valid_file_paths, valid_labels

# Define transforms with time and frequency masking
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    # T.TimeMasking(time_mask_param=80),
    # T.FrequencyMasking(freq_mask_param=30)
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dir = r"train_log_spec"
# test_dir = r"D:\Music Classification\test"
val_dir = r"test_log_spec"

train_dataset = SpectrogramDataset(train_dir, transform=train_transform)
# test_dataset = SpectrogramDataset(test_dir, transform=val_test_transform)
val_dataset = SpectrogramDataset(val_dir, transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
