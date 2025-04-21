import os
import sys
import time
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType
from pyspark.sql.functions import rand
from conf import pytorch_model_state_file, images_path_csv, images_dir, model_lable_map, predication_detail_query, confusion_matrix_query, onnx_model_state_file, default_backend

backend = default_backend
if len(sys.argv) > 1:
    backend = sys.argv[1]

def predict_image(image_path, backend="pytorch"):
    """
    Lazy load the model once per worker based on backend type,
    then run inference on image_path.
    
    :param image_path: Path to the image.
    :param backend: 'pytorch' or 'onnx' (default: 'pytorch')
    :return: Prediction output (can be a class id or dict with details).
    """
    if backend == "onnx":
        if not hasattr(predict_image, "onnx_model"):
            from onnx_scripts.inference import ONNXModel
            print(f"Loading onnx model from: {onnx_model_state_file}")
            predict_image.onnx_model = ONNXModel(onnx_model_state_file)
        return predict_image.onnx_model.predict(image_path=image_path)

    elif backend == "pytorch":
        if not hasattr(predict_image, "pytorch_model"):
            from pytorch_scripts.inference import ModelHandler
            print(f"Loading pytorch model from: {pytorch_model_state_file}")
            predict_image.pytorch_model = ModelHandler(pytorch_model_state_file)
        return predict_image.pytorch_model.predict(image_path)

    else:
        raise ValueError(f"Unsupported backend: {backend}")



def get_image_inference_udf(image_path, backend="pytorch"):
    """
    Lazy load the PyTorch model once in each executor (worker).
    Then run inference and map the numeric class to a string label.
    """
    
    pred_details = predict_image(image_path, backend)


    # Get numeric prediction (e.g. 0, 1, or 2).
    pred_class_id = int(pred_details.get("predicted_class",-1))
    pred_label  = str(model_lable_map.get(pred_class_id, "Unknown"))
    pred_class_confidence = float(pred_details.get("confidence", -1))
    class_0_prob = float(pred_details.get("probabilities",[])[0])
    class_1_prob = float(pred_details.get("probabilities",[])[1])
    class_2_prob = float(pred_details.get("probabilities", [])[2])

    # if not get_details:
    #     return pred_class, pred_label

    # Return with details
    return pred_label, pred_class_id, pred_class_confidence, class_0_prob, class_1_prob, class_2_prob


# Start Spark Session
spark = SparkSession.builder.appName("ImagePrediction").getOrCreate()


schema = StructType(
    [
        StructField("pred_label", StringType(), True),
        StructField("pred_class_id", IntegerType(), True),
        StructField("pred_class_confidence", FloatType(), True),
        StructField("class_0_prob", FloatType(), True),
        StructField("class_1_prob", FloatType(), True),
        StructField("class_2_prob", FloatType(), True),
    ]
)

spark.udf.register(
    "get_image_inference_udf",
    lambda path, backend: get_image_inference_udf(os.path.join(images_dir, path), backend),
    schema,
)

# Load images DataFrame (assuming it has a column 'image_path')
image_df = spark.read.csv(images_path_csv, header=True, inferSchema=True)

# Create a temporary view to apply SQL
image_df.createOrReplaceTempView("images")

start = time.time()
predictions_df = spark.sql(predication_detail_query.format(backend=backend))
print(f"Prediction Details Query Time: {time.time() - start}")

start = time.time()
confusion_ma_df = spark.sql(confusion_matrix_query.format(backend=backend))
print(f"Confusion Matrix Query Time: {time.time() - start}")

# Show results
predictions_df.orderBy(rand()).show(truncate=False, n=30)
predictions_df.coalesce(1).write.mode("overwrite").option("header", True).csv("output_predictions")

confusion_ma_df.show()

spark.stop()
