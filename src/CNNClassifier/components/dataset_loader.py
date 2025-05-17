import torch
from torch.utils.data import DataLoader
from typing import Optional
from pathlib import Path
import pandas as pd
from CNNClassifier.components.dataset import ImageDataset
from CNNClassifier.logger import logger


class DatasetLoader:
    def __init__(self, 
                 dataset_path: Path,
                 shuffle: bool = False,
                 name: str = "",
                 batch_size: Optional[int] = None,
                 num_workers: Optional[int] = 4):
        self.dataset_path = dataset_path
        if name == "train":
            self.batch_size = batch_size
        else:
            bs = pd.read_csv(str(self.dataset_path)).shape[0]
            self.batch_size = bs
        self.shuffle = shuffle
        self.name = name
        self.num_workers = num_workers
        if self.dataset_path.suffix == ".pt":
            self._load_dataset(self.dataset_path)
        else:
            self._create_dataset()
            self._save_dataset()
    
    def _create_dataset(self) -> None:
        try:
            self.dataset = ImageDataset(data_path = self.dataset_path,
                                        images_path = self.dataset_path.parent / "images")
            self.data_loader = DataLoader(
                dataset=self.dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers
            )
            
            logger.info(f"Created data loaders for {self.dataset_path.stem} dataset")
        except Exception as e:
            logger.error(f"Error creating data loader for {self.dataset_path.stem} dataset: {e}")
            raise e
            
    def _save_dataset(self) -> None:
        try:
            save_path = self.dataset_path.parent / f"{self.dataset_path.stem}.pt"
            torch.save(self.data_loader.dataset, save_path)
            logger.info(f"Dataset saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving dataset to {save_path}: {e}")
            raise e
        
    def _load_dataset(self, dataset_path: Path) -> None:
        try:
            dataset = torch.load(dataset_path)
            self.data_loader = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers
            )
            logger.info("Successfully loaded dataset & created data loader")
        except Exception as e:
            logger.error(f"Error loading dataset & creating data loader: {e}")
            raise e