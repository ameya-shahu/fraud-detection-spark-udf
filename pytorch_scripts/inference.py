import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model.model import CustomCNN  # Your model definition

# Preprocessing setup (match training transforms)
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class ModelHandler:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        model = CustomCNN(num_classes=3)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        return model
    
    def get_classes_probabilities(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)  
            return probabilities


    def predict(self, image_path: str):
        probabilities = self.get_classes_probabilities(image_path)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

        return {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "probabilities": probabilities[0].tolist()
        }
