import torch
from torch.utils.data import DataLoader, random_split
from typing import Tuple
from .dataset import ImageDataset
from ..config.config import DataLoaderConfig


class DataLoaderFactory:
    @staticmethod
    def create_data_loaders(num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
        full_dataset = ImageDataset(data_path = DataLoaderConfig.data_path,
                                    images_path = DataLoaderConfig.images_path)
        train_size = int(DataLoaderConfig.train_split * len(full_dataset))
        val_size = int(DataLoaderConfig.val_split) * len(full_dataset)
        test_size = len(full_dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, 
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42) 
        )
        
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=DataLoaderConfig.batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=len(val_dataset),
            shuffle=False,
            num_workers=num_workers
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=len(test_dataset),
            shuffle=False,
            num_workers=num_workers
        )
        
        return train_loader, val_loader, test_loader