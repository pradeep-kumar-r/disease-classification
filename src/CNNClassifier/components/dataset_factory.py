import torch
from torch.utils.data import DataLoader, random_split
from typing import Tuple
from CNNClassifier.components.dataset import ImageDataset
from CNNClassifier.config import DataLoaderConfig
from CNNClassifier.logger import logger


class DatasetFactory:
    @staticmethod
    def get_datasets(num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
        try:
            full_dataset = ImageDataset(data_path = DataLoaderConfig.data_folder_path,
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
            
            logger.info("Created data loaders for train, validation and test sets")
            return train_loader, val_loader, test_loader
        except Exception as e:
            logger.error(f"Error creating data loaders: {e}")
            raise e