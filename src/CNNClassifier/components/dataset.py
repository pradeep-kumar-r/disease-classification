import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import Tuple, Optional
import os
import pandas as pd
from CNNClassifier.logger import logger


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
        
        self.data = []
        self.classes = set()
        data_df = pd.read_csv(data_path)
        for _, row in data_df.iterrows():
            image_name = row['images']
            label = row['label']
            self.data.append((image_name, label))
            self.classes.add(label)
        
        self.classes = sorted(list(self.classes))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.data = [(img_name, self.class_to_idx[label]) for img_name, label in self.data]
        logger.info(f"Loaded dataset with {len(self.data)} images and {len(self.classes)} classes")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if idx >= self.__len__():
            logger.error(f"Index {idx} is out of range")
            raise IndexError(f"Index {idx} is out of range")
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name, label = self.data[idx]
        img_path = os.path.join(self.images_path, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            return image, label
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            raise e
