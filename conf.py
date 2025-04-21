import os

# Define the project root folder path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")

images_dir = os.path.join(DATASET_DIR, "all_images")
images_path_csv = os.path.join(DATASET_DIR, "image_data.csv")

onnx_model_state_file_name = 'id_classification_cnn_model.onnx'
pytorch_model_state_file_name = 'id_classification_cnn_model.pth'

onnx_model_state_file = os.path.join(MODEL_DIR, onnx_model_state_file_name)
pytorch_model_state_file = os.path.join(MODEL_DIR, pytorch_model_state_file_name)

default_backend = "pytorch"


model_lable_map = {
        0: "crop_and_replace",
        1: "font_issue",
        2: "Valid Id"
    }


predication_detail_query = """
    SELECT 
        image_path,
        actual_class,
        pred.pred_label,
        actual_class_id,
        pred.pred_class_id,
        ROUND(pred.pred_class_confidence, 2) AS pred_class_confidence,
        ROUND(pred.class_0_prob, 2) AS class_0_prob,
        ROUND(pred.class_1_prob, 2) AS class_1_prob,
        ROUND(pred.class_2_prob, 2) AS class_2_prob
    FROM (
        SELECT 
            image_path,
            actual_class,
            actual_class_id,
            get_image_inference_udf(image_path, '{backend}') AS pred
        FROM images
    ) AS predictions
    """

confusion_matrix_query = """
SELECT
    SUM(CASE WHEN actual_class_id IN (0, 1) AND pred.pred_class_id IN (0, 1) THEN 1 ELSE 0 END) AS true_positives,
    SUM(CASE WHEN actual_class_id = 2 AND pred.pred_class_id IN (0, 1) THEN 1 ELSE 0 END) AS false_positives,
    SUM(CASE WHEN actual_class_id = 2 AND pred.pred_class_id = 2 THEN 1 ELSE 0 END) AS true_negatives,
    SUM(CASE WHEN actual_class_id IN (0, 1) AND pred.pred_class_id = 2 THEN 1 ELSE 0 END) AS false_negatives
FROM (
    SELECT 
        image_path,
        actual_class,
        actual_class_id,
        get_image_inference_udf(image_path, '{backend}') AS pred
    FROM images
) AS predictions
"""