import os
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import decode_image


class FloodNetDataset(Dataset):
    def __init__(self, label_dir, image_dir, transform=None, target_transform=None, fine_tune=False):
        self.label_dir = label_dir
        self.image_dir = image_dir
        self.fine_tune = fine_tune

        # Get the list of labels and images and sort them in order to match them
        self.labels = list(sorted(os.listdir(self.label_dir)))
        self.images = list(sorted(os.listdir(self.image_dir)))

        if len(self.labels) != len(self.images):
            raise ValueError("Number of labels and images do not match")

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.labels[idx])
        
        if not self.fine_tune:
            image = decode_image(img_path).type(torch.float32) / 255.0
            label = decode_image(label_path)

            # Resize image and label to 512x512
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0), size=(512, 512), mode="bilinear", align_corners=False
            ).squeeze(0)
            label = torch.nn.functional.interpolate(
                label.unsqueeze(0), size=(512, 512), mode="nearest"
            ).squeeze(0)
            
            if label.shape[0] > 1:
                label = label[0, :, :]
                label = label.unsqueeze(0)
        
            label = label.type(torch.LongTensor)
        else:
            image = Image.open(img_path).convert('RGB')
            # Load mask and ensure it's in the correct format
            label = Image.open(label_path)
            label = np.array(label)
        
            # Verify mask dimensions
            if len(label.shape) > 2:
                label = label[:,:,0]  # Take first channel if multiple channels exist
                
            # Resize mask to match image size if needed
            label = Image.fromarray(label)
            label = label.resize((512, 512), Image.NEAREST)
            label = np.array(label)
            label = torch.from_numpy(label).long()

        
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
            
        return image, label
