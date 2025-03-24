import torch
from torch.utils.data import Dataset
import albumentations as A
import numpy as np
import json
from PIL import Image
from utils import get_targets

class COCODatasetWrapper(Dataset):
    def __init__(self, hf_dataset, transforms=A.Normalize(), output_size=(128, 128), num_classes=80):
        self.hf_dataset = hf_dataset.with_format("numpy")
        self.transforms = transforms
        self.output_size = output_size
        self.num_classes = num_classes
    
    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        # Get image and annotations
        sample = self.hf_dataset[idx]
        
        # Convert PIL image to numpy array
        image = np.array(sample['image'])
        
        # Get bounding boxes and labels from the objects dictionary
        boxes = np.array(sample['objects']['bbox'], dtype=np.float32)
        labels = np.array(sample['objects']['category'], dtype=np.int64)

        # Normalize bounding box coordinates
        transformed = self.transforms(image=image, bboxes=boxes, labels=labels)

        image = transformed['image']
        boxes = np.array(transformed['bboxes'], dtype=np.float32)
        labels = np.array(transformed['labels'])

        height, width = image.shape[:2]
        boxes[:, [0, 2]] /= width
        boxes[:, [1, 3]] /= height

        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1)  # Convert to CxHxW format
        boxes = torch.from_numpy(boxes)
        labels = torch.from_numpy(labels)

        # Generate CenterNet targets
        targets = get_targets(boxes, labels, self.output_size, self.num_classes)

        return {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'heatmap': targets['heatmap'],
            'wh': targets['wh'],
            'offset': targets['offset'],
            'reg_mask': targets['reg_mask']
        }