import torch
from model import CustomCNN

# Instantiate and prepare the model
model = CustomCNN(num_classes=3)
model.initialize_fc()  # Initialize FC layer
model.load_state_dict(torch.load("id_classification_cnn_model.pth"))
model.eval()

# Create dummy input for ONNX tracing
dummy_input = torch.randn(1, 3, 128, 128)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11,
)

print("✅ Exported model to model.onnx")
