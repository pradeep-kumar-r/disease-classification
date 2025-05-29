from pathlib import Path
from typing import Literal, Optional
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from CNNClassifier.logger import logger
from CNNClassifier.components.dataset_loader import DatasetLoader


class ModelEvaluator:
    def __init__(self, 
                 model: nn.Module,
                 dataloader: DatasetLoader, 
                 criterion: nn.Module=nn.CrossEntropyLoss(),
                 report_save_path: Optional[Path] = None,
                 device: Literal["cuda", "cpu"]='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = model
        self.model.to(self.device)
        self.dataloader = dataloader
        self.class_names = list(self.dataloader.class_to_idx.keys())
        self.criterion = criterion
        self.report_save_path = report_save_path
        self.confusion_matrix = None
        self.classification_report = None
        self.accuracy = 0.0
        self.predictions = np.array([])
        self.labels = np.array([])
        self.probabilities = np.array([])
        self.is_evaluated = False
        
    def evaluate(self):
        if not self.is_evaluated:
            self.model.eval()
            with torch.no_grad():
                for input_data, label in self.dataloader:
                    input_data, label = input_data.to(self.device), label.to(self.device)
                    self.labels = np.append(self.labels, label.numpy())
                    outputs = self.model.forward(input_data)
                    probabilities = F.softmax(outputs, dim=1)
                    predicted = probabilities.argmax(dim=1)
                    self.probabilities = np.append(self.probabilities, probabilities.numpy())
                    self.predictions = np.append(self.predictions, predicted.numpy())
                    
            self.accuracy = accuracy_score(self.labels, self.predictions)
            self.confusion_matrix = confusion_matrix(self.labels, 
                                                     self.predictions, 
                                                     labels=self.class_names)
            self.classification_report = classification_report(self.labels, 
                                                               self.predictions,
                                                               labels=self.class_names,
                                                               target_names=self.class_names)
            logger.info("Evaluation"
                        f"Acc: {self.accuracy:.4f}"
                        f"Confusion Matrix: \n{self.confusion_matrix}"
                        f"Classification Report: \n{self.classification_report}"
                        )
            self.is_evaluated = True
        
    def save_report(self):
        if self.report_save_path:
            with open(self.report_save_path, 'w', encoding='utf-8') as f:
                f.write("Evaluation Report:\n\n\n")
                f.write(f"Accuracy: {self.accuracy:.4f}\n\n\n")
                f.write(f"Confusion Matrix: \n{self.confusion_matrix}\n\n\n")
                f.write(f"Classification Report: \n{self.classification_report}\n")
            logger.info(f"Report saved to {self.report_save_path}")
        else:
            logger.info("Report save path not specified")
            raise FileNotFoundError("Report save path not specified")
        