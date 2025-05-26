from pathlib import Path
from typing import Literal, Optional
import torch
from torch import nn as nn
from CNNClassifier.logger import logger
from CNNClassifier.components.dataset_loader import DatasetLoader


class ModelEvaluator:
    def __init__(self, 
                 model: nn.Module,
                 test_dataloader: DatasetLoader, 
                 criterion: nn.Module=nn.CrossEntropyLoss(),
                 report_save_path: Optional[Path] = None,
                 device: Literal["cuda", "cpu"]='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = model
        self.model.to(self.device)
        self.test_dataloader = test_dataloader
        self.criterion = criterion
        self.report_save_path = report_save_path
        self.test_loss = float("inf")
        self.test_acc = 0.0
        
    def evaluate(self):
        self.model.eval()
        loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.test_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                l = self.criterion(outputs, labels)

                loss += l.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        self.avg_test_loss = loss / len(self.test_dataloader)
        self.test_acc = correct / total
        logger.info(f"Test Loss: {loss:.4f}, Avg Loss: {self.avg_loss:.4f}, Test Acc: {self.test_acc:.4f}")
        
    def save_report(self):
        if self.report_save_path:
            with open(self.report_save_path, 'w') as f:
                f.write(f"Test Loss: {self.avg_test_loss:.4f}\n")
                f.write(f"Test Acc: {self.test_acc:.4f}\n")
            logger.info(f"Report saved to {self.report_save_path}")
        else:
            logger.info("Report save path not specified")
            raise FileNotFoundError("Report save path not specified")
        