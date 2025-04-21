import torch
from model import CustomCNN
from conf import pytorch_model_state_file_name, onnx_model_state_file_name

# Instantiate and prepare the model
model = CustomCNN(num_classes=3)
model.initialize_fc()  # Initialize FC layer
model.load_state_dict(torch.load(pytorch_model_state_file_name))
model.eval()

# Create dummy input for ONNX tracing
dummy_input = torch.randn(1, 3, 128, 128)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    onnx_model_state_file_name,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11,
)

print("✅ Exported model to model.onnx")
