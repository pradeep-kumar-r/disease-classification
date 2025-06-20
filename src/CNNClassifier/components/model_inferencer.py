from pathlib import Path
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Literal, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from CNNClassifier.logger import logger
from CNNClassifier.components.model import BasicCNNModel


class ModelInferencer:
    def __init__(self,
                 model_path: Path,
                 device: Literal["cuda", "cpu"] = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model: Optional[nn.Module] = None
        self.model_metadata: Optional[Dict] = None
        self.class_names: List[str] = []
        self.model_path: Path = model_path
        self._load_model()
        
    def _load_model(self) -> None:
        try:
            checkpoint = torch.load(self.model_path, 
                                    map_location=self.device, 
                                    weights_only=False
                                    )
            num_classes = len(checkpoint['train_data_metadata']['class_to_idx'].keys())
            self.model = BasicCNNModel(num_classes=num_classes)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            self.class_to_idx = checkpoint['train_data_metadata']['class_to_idx']
            self.class_names = list(self.class_to_idx.keys())
            logger.info(f"Loaded model from {self.model_path}")
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise

    def _preprocess_image(self, input_image: Image) -> torch.Tensor:
        try:
            image = input_image.convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            image_tensor = transform(image)
            image_tensor = image_tensor.unsqueeze(0)
            return image_tensor.to(self.device)
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise

    def predict(self, input_data: Union[Image.Image, bytes]) -> Tuple[str, float]:
        try:
            if not self.model:
                raise ValueError("Model not loaded. Please load the model first.")
                
            if isinstance(input_data, bytes):
                input_image = Image.open(BytesIO(input_data))
            else:
                input_image = input_data
                
            input_tensor = self._preprocess_image(input_image)
            with torch.no_grad():
                outputs = self.model.forward(input_tensor).to(self.device)
                probabilities = F.softmax(outputs, dim=1)
                predicted = probabilities.argmax(dim=1)
                confidence = probabilities[0][predicted.item()].item()
                predicted_class = self.class_names[predicted.item()]
                probabilities_dict = {class_name: probabilities[0][i].item() for i, class_name in enumerate(self.class_names)}
                return predicted_class, confidence, probabilities_dict
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise