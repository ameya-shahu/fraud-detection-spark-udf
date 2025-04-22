import torch
from model import CustomCNN

# filenames
pytorch_model_state_file_name = "E:\Files\ASU\sem_4\disml_final_project\spark-udf-cnn-inference\model\id_classification_cnn_model.pth"
onnx_model_state_file_name = "id_classification_cnn_model.onnx"

# 1. Instantiate your model (matches how you trained it)
model = CustomCNN(num_classes=3)

# 2. Load the weights (use map_location if needed)
state = torch.load(pytorch_model_state_file_name, map_location="cpu")
model.load_state_dict(state)
model.eval()

# 3. Create a dummy input with the same shape you trained on
dummy_input = torch.randn(1, 3, 128, 128)

# 4. Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    onnx_model_state_file_name,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11,
)

print(f"✅ Exported model to {onnx_model_state_file_name}")
