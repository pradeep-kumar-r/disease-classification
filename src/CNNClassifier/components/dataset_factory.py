import torch
from torch.utils.data import DataLoader, random_split
from typing import Tuple, Dict
import json
from pathlib import Path
from CNNClassifier.components.dataset import ImageDataset
from CNNClassifier.config import DataLoaderConfig, ArtefactsConfig
from CNNClassifier.logger import logger


class DatasetFactory:
    def __init__(self):
        self.dataset_path = Path(ArtefactsConfig.artefacts_path) / "datasets"
        self.metadata_path = self.dataset_path / "metadata.json"
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        
    def get_datasets(self, num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
        try:
            full_dataset = ImageDataset(data_path = DataLoaderConfig.dataset_path,
                                      images_path = DataLoaderConfig.images_path)
            train_size = int(DataLoaderConfig.train_split * len(full_dataset))
            val_size = int(DataLoaderConfig.val_split * len(full_dataset))
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
            
    def save_datasets(self) -> None:
        """Save the datasets and their metadata"""
        try:
            # Save datasets
            torch.save(self.train_loader.dataset, self.dataset_path / "train_dataset.pt")
            torch.save(self.val_loader.dataset, self.dataset_path / "val_dataset.pt")
            torch.save(self.test_loader.dataset, self.dataset_path / "test_dataset.pt")
            
            # Save metadata
            metadata = {
                "train_size": len(self.train_loader.dataset),
                "val_size": len(self.val_loader.dataset),
                "test_size": len(self.test_loader.dataset),
                "batch_size": DataLoaderConfig.batch_size,
                "train_split": DataLoaderConfig.train_split,
                "val_split": DataLoaderConfig.val_split,
                "num_workers": self.train_loader.num_workers
            }
            
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
                
            logger.info(f"Datasets saved to {self.dataset_path}")
            
        except Exception as e:
            logger.error(f"Error saving datasets: {e}")
            raise e
            
    def load_datasets(self, num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load the saved datasets"""
        try:
            # Load datasets
            train_dataset = torch.load(self.dataset_path / "train_dataset.pt")
            val_dataset = torch.load(self.dataset_path / "val_dataset.pt")
            test_dataset = torch.load(self.dataset_path / "test_dataset.pt")
            
            # Create dataloaders
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
            
            logger.info("Loaded saved datasets successfully")
            return train_loader, val_loader, test_loader
            
        except Exception as e:
            logger.error(f"Error loading datasets: {e}")
            raise e
            
    def get_metadata(self) -> Dict:
        """Get the metadata of saved datasets"""
        try:
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            raise e