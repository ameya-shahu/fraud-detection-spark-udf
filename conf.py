import os

# Define the project root folder path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")

images_dir = os.path.join(DATASET_DIR, "all_images")
images_path_csv = os.path.join(DATASET_DIR, "image_names.csv")

onnx_model_state_file_name = 'id_classification_cnn_model.onnx'
pytorch_model_state_file_name = 'id_classification_cnn_model.pth'

onnx_model_state_file = os.path.join(MODEL_DIR, onnx_model_state_file_name)
pytorch_model_state_file = os.path.join(MODEL_DIR, pytorch_model_state_file_name)


model_lable_map = {
        0: "crop_and_replace",
        1: "font_issue",
        2: "Valid Id"
    }

