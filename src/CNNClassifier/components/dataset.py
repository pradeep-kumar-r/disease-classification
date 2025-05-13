import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import Tuple, Optional
import os

from ..logger import logger


class ImageDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        images_path: str,
        transform: Optional[transforms.Compose] = None
    ):
        self.data_path = data_path
        self.images_path = images_path
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        self.data = {}
        with open(data_path, "r") as file:
            for line in file:
                image_name, label = line.strip().split(",")
                if label != "label":
                    self.data[image_name] = label
                
        self.classes = list(set(self.data.values()))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        for image_name, label in self.data.items():
            self.data[image_name] = self.class_to_idx[label]
        self.data = list(self.data.items())
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if idx >= self.__len__():
            logger.error(f"Index {idx} is out of range")
            raise IndexError(f"Index {idx} is out of range")
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_path, label = self.data[idx]
        img_path = os.path.join(self.images_path, img_path)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, label
