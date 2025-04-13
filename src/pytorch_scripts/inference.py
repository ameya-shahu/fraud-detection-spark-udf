import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model import CustomCNN  # Your model definition

# Preprocessing setup (match training transforms)
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class ModelHandler:
    _model = None

    @classmethod
    def load_model(cls, model_path="id_classification_cnn_model.pth"):
        if cls._model is None:
            model = CustomCNN(num_classes=3)
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.eval()
            cls._model = model
        return cls._model

    @staticmethod
    def predict(image_path: str):
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        model = ModelHandler.load_model()
        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
        return predicted_class
