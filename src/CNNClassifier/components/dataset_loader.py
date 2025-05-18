from typing import Optional
from torch.utils.data import DataLoader
from CNNClassifier.components.image_dataset import ImageDataset
from CNNClassifier.logger import logger


class DatasetLoader():
    def __init__(self,
                 dataset: ImageDataset,
                 shuffle: Optional[bool] = None,
                 batch_size: Optional[int] = None,
                 num_workers: Optional[int] = None):
        self.dataset = dataset
        if dataset.dataset_type == "train":
            self.batch_size = batch_size
            self.shuffle = True if shuffle is None else shuffle
        else:
            self.batch_size = dataset.__len__()
            self.shuffle = False if shuffle is None else shuffle
        self.num_workers = 4 if num_workers is None else num_workers
        self.dataloader = self._create_dataloader()
    
    def _create_dataloader(self) -> None:
        try:
            dataloader = DataLoader(
                dataset=self.dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers
            )
            logger.info(f"Successfully created dataloader"
                        f"for {self.dataset.dataset_type} dataset"
                        f"with {self.dataset.__len__()} samples and {self.dataset.num_classes()} classes"
                        f"with {self.batch_size} batch size"
                        f"and {self.num_workers} num workers")
            return dataloader
        except Exception as e:
            logger.error(f"Error created dataloader: {e}")
            raise e