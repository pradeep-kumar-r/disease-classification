import torch
from tqdm import tqdm
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
from typing import Literal
from CNNClassifier.logger import logger
from CNNClassifier.config import ModelTrainingConfig
from CNNClassifier.components.model import NNModel
from CNNClassifier.components.dataset_factory import DatasetFactory


class ModelTrainer:
    def __init__(self, 
                 model: NNModel, 
                 datasets: DatasetFactory, 
                 num_epochs: int=ModelTrainingConfig.num_epochs, 
                 criterion: nn.Module=nn.CrossEntropyLoss(),
                 learning_rate: float=ModelTrainingConfig.learning_rate, 
                 device: Literal["cuda", "cpu"]='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = model.to(self.device)
        self.datasets = datasets
        self.train_dataloader, self.val_dataloader, self.test_dataloader = self.datasets.get_datasets()
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
    def _train_step(self, epoch):
        self.model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_dataloader, desc=f'Epoch {epoch+1}/{self.num_epochs}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar with current batch metrics
            pbar.set_postfix({
                'Loss': epoch_loss/total,
                'Accuracy': 100.*correct/total
            })

        # Calculate final metrics for the epoch
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

        # Calculate final validation metrics
        avg_val_loss = val_loss / len(self.val_dataloader)
        val_accuracy = correct / total
        
        return avg_val_loss, val_accuracy

    def train(self, save_path: Path):
        best_val_acc = 0.0
        
        for epoch in range(self.num_epochs):
            # Training phase
            train_loss, train_acc = self._train_step(epoch)
            
            # Validation phase
            val_loss, val_acc = self._validate()
            
            # Log metrics
            logger.info(
                f'Epoch {epoch+1}/{self.num_epochs} - '
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}'
            )

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, save_path)
                logger.info(f'Model saved with validation accuracy: {val_acc:.4f}')
            
            # Clear GPU memory if using CUDA
            if self.device == 'cuda':
                torch.cuda.empty_cache()

    def load_model(self, model_path: Path):
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['val_acc']