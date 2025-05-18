import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Literal
import os
import pandas as pd
from CNNClassifier.logger import logger


class ImageDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        images_path: str,
        transform: Optional[transforms.Compose] = None,
        dataset_type: Literal["train", "val", "test"] = "train"
    ):
        self.data_path = data_path
        self.images_path = images_path
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        self.dataset_type = dataset_type
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
        logger.info(f"Created dataset with {len(self.data)} images and {len(self.classes)} classes")
    
    def num_classes(self) -> int:
        return len(self.classes)
    
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
        
    def export_dataset(self, save_path: Path) -> None:
        try:
            metadata = {
                'data': self.data,  # List of (image_name, label) tuples
                'data_path': self.data_path,
                'images_path': self.images_path,
                'transform': self.transform,
                'class_to_idx': self.class_to_idx,
                'dataset_type': self.dataset_type
            }
            
            torch.save(metadata, save_path)
            logger.info(f"Dataset metadata exported to {save_path}")
        except Exception as e:
            logger.error(f"Error exporting dataset metadata: {e}")
            raise e

    def __eq__(self, other: 'ImageDataset') -> bool:
        return self.data == other.data and self.images_path == other.images_path and self.data_path == other.data_path and self.dataset_type == other.dataset_type and self.num_classes() == other.num_classes() and self.__len__() == other.__len__()
     
    @classmethod
    def load_dataset(cls, metadata_path: Path) -> 'ImageDataset':
        try:
            metadata = torch.load(metadata_path, weights_only=False)
            dataset = cls(
                data_path=metadata['data_path'],
                images_path=metadata['images_path'],
                dataset_type=metadata['dataset_type']
            )
            dataset.data = metadata['data']
            # dataset.class_to_idx = metadata['class_to_idx']
            logger.info(f"Load dataset from {metadata_path} with {len(dataset)} images and {dataset.num_classes()} classes")
            return dataset
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise e
