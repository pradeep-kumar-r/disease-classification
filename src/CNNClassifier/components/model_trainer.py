from typing import Literal, Optional
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from CNNClassifier.logger import logger
from CNNClassifier.components.dataset_loader import DatasetLoader


class ModelTrainer:
    def __init__(self, 
                 model: nn.Module,
                 train_dataloader: DatasetLoader,
                 val_dataloader: DatasetLoader,
                 num_epochs: int,
                 learning_rate: float,
                 criterion: nn.Module=nn.CrossEntropyLoss(),
                 model_save_path: Optional[Path] = None,
                 device: Literal["cuda", "cpu"]='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device: Literal["cuda", "cpu"] = device
        self.model: nn.Module = model.to(self.device)
        self.train_dataloader: DatasetLoader = train_dataloader
        self.val_dataloader: DatasetLoader = val_dataloader
        self.learning_rate: float = learning_rate
        self.num_epochs: int = num_epochs
        self.criterion: nn.Module = criterion
        self.train_acc: float = 0.0
        self.val_acc: float = 0.0
        self.train_loss: float = float("inf")
        self.val_loss: float = float("inf")
        self.best_val_acc: float = 0.0
        self.model_save_path: Optional[Path] = model_save_path
        self.optimizer: optim.Optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
    def _train_step(self, epoch):
        self.model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_dataloader, desc=f'Epoch {epoch+1}/{self.num_epochs}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model.forward(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'Loss': epoch_loss/total,
                'Accuracy': 100.*correct/total
            })

        avg_loss = epoch_loss / len(self.train_dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy

    def _validate(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.val_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_val_loss = val_loss / len(self.val_dataloader)
        val_accuracy = correct / total
        
        return avg_val_loss, val_accuracy

    def _save_model(self, epoch: int) -> None:
        train_data_metadata = {
                'data_path': self.train_dataloader.dataset.data_path,
                'images_path': self.train_dataloader.dataset.images_path,
                'transform': self.train_dataloader.dataset.transform,
                'class_to_idx': self.train_dataloader.dataset.class_to_idx,
                'dataset_type': self.train_dataloader.dataset.dataset_type
            }
        val_data_metadata = {
                'data_path': self.val_dataloader.dataset.data_path,
                'images_path': self.val_dataloader.dataset.images_path,
                'transform': self.val_dataloader.dataset.transform,
                'class_to_idx': self.val_dataloader.dataset.class_to_idx,
                'dataset_type': self.val_dataloader.dataset.dataset_type
            }
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_acc': self.train_acc,
            'val_acc': self.val_acc,
            'train_data_metadata': train_data_metadata,
            'val_data_metadata': val_data_metadata
        }, self.model_save_path)
        logger.info(f'Model saved.\nTrain accuracy: {self.train_acc:.4f}\nValidation accuracy: {self.val_acc:.4f}')
    
    def __str__(self):
        st = ""
        st += f"\nModel Path: {self.model_save_path or 'TBD'}\n"
        st += f"Number of epochs: {self.num_epochs}\n"
        st += f"Learning rate: {self.learning_rate}\n"
        st += f"Device: {self.device}\n"
        st += "Model metrics:\n"
        st += f"Train accuracy: {self.train_acc:.4f}\n"
        st += f"Train loss: {self.train_loss:.4f}\n"
        st += f"Validation accuracy: {self.val_acc:.4f}\n"
        st += f"Validation loss: {self.val_loss:.4f}\n"
        return st
    
    def train(self) -> None:
        logger.info(
            f"Starting Training\n"
            f"Number of epochs: {self.num_epochs}\n"
            f"Learning rate: {self.learning_rate}\n"
            f"Device: {self.device}\n"
            f"Train accuracy: {self.train_acc:.4f}\n"
            f"Train loss: {self.train_loss:.4f}\n"
            f"Validation accuracy: {self.val_acc:.4f}\n"
            f"Validation loss: {self.val_loss:.4f}\n"
        )
        
        for epoch in range(self.num_epochs):
            train_loss, train_acc = self._train_step(epoch)
            val_loss, val_acc = self._validate()
            
            logger.info(
                f'Epoch {epoch+1}/{self.num_epochs} - '
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}'
            )
                
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        
        if val_acc > self.val_acc:
            if self.model_save_path:
                self._save_model(epoch)
            self.best_val_acc = val_acc
            
    def load_model(self, model_path: Path) -> None:
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_acc = checkpoint['train_acc']
        self.val_acc = checkpoint['val_acc']
        logger.info(f'Model loaded.\nTrain accuracy: {self.train_acc:.4f}\nValidation accuracy: {self.val_acc:.4f}')