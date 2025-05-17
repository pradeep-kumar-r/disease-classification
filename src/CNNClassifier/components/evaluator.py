import torch
from pathlib import Path
from typing import Literal
from CNNClassifier.logger import logger
from CNNClassifier.components.dataset_loader import DatasetFactory


class ModelEvaluator:
    def __init__(self, 
                 model_path: Path,
                 datasets: DatasetFactory, 
                 device: Literal["cuda", "cpu"]='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self._load_model(model_path)
        self.model.to(self.device)
        _, _, self.test_dataloader = datasets.get_datasets()
        
    def evaluate(self):
        self.model.eval()
        loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.val_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                l = self.criterion(outputs, labels)

                loss += l.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        # Calculate final validation metrics
        avg_loss = loss / len(self.val_dataloader)
        accuracy = correct / total
        logger.info(f"Test Loss: {avg_loss:.4f}, Test Acc: {accuracy:.4f}")
        
        return avg_loss, accuracy

    def _load_model(self, model_path: Path):
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Model loaded from {model_path}")