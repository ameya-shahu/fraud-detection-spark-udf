import numpy as np
import onnxruntime
from PIL import Image
from torchvision import transforms

# Define preprocessing (same as training)
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class ONNXModel:
    def __init__(self, model_path="model.onnx"):
        self.session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, image_path):
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).numpy()  # Convert to Numpy batch
        return tensor

    

    def predict(self, image_path):
        input_data = self.preprocess(image_path)
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        logits = outputs[0]  # Shape: (1, num_classes)

        # Apply softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        predicted_class = int(np.argmax(probabilities, axis=1)[0])
        confidence = float(probabilities[0][predicted_class])

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities[0].tolist()
    }
