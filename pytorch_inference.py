import os
import time
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType

from conf import pytorch_model_state_file, images_path_csv, images_dir, model_lable_map


def predict_image(image_path):
    """
    Lazy load the PyTorch model once in each executor (worker).
    Then run inference and map the numeric class to a string label.
    """
    if not hasattr(predict_image, "model_handler"):
        from pytorch_scripts.inference import ModelHandler

        print(f"Loading model from: {pytorch_model_state_file}")
        predict_image.model_handler = ModelHandler(pytorch_model_state_file)

    # Get numeric prediction (e.g. 0, 1, or 2).
    numeric_pred = int(predict_image.model_handler.predict(image_path))

    # Return the string label (or "Unknown" if out of range).
    return model_lable_map.get(numeric_pred, "Unknown")


def predict_image_with_confidence(image_path):
    """
    Lazy load the PyTorch model once in each executor (worker).
    Then run inference and map the numeric class to a string label.
    """
    if not hasattr(predict_image, "model_handler"):
        from pytorch_scripts.inference import ModelHandler

        print(f"Loading model from: {pytorch_model_state_file}")
        predict_image.model_handler = ModelHandler(pytorch_model_state_file)

    # Get numeric prediction (e.g. 0, 1, or 2).
    numeric_pred, confidence_score = predict_image.model_handler.predict(
        image_path, show_confidence=True
    )
    numeric_pred = int(numeric_pred)
    label = model_lable_map.get(numeric_pred, "Unknown")

    return (label, float(confidence_score))


# Start Spark Session
spark = SparkSession.builder.appName("ImagePrediction").getOrCreate()

# Register your Python function as a UDF returning String
spark.udf.register(
    "predict_image_udf",
    lambda path: predict_image(os.path.join(images_dir, path)),
    StringType(),
)

schema = StructType(
    [
        StructField("label", StringType(), True),
        StructField("confidence", FloatType(), True),
    ]
)
spark.udf.register(
    "predict_image_with_confidence_udf",
    lambda path: predict_image_with_confidence(os.path.join(images_dir, path)),
    schema,
)

# Load images DataFrame (assuming it has a column 'image_path')
image_df = spark.read.csv(images_path_csv, header=True, inferSchema=True)

# Create a temporary view to apply SQL
image_df.createOrReplaceTempView("images")

start = time.time()
# Use Spark SQL to call the UDF
predictions_df = spark.sql(
    """
    SELECT
        image_path,
        predict_image_udf(image_path) AS predicted_class
    FROM images
"""
)

print(f"query run time: {time.time() - start}")

start = time.time()

predictions_df_confidence = spark.sql(
    """
    SELECT
        image_path,
        pred.label AS predicted_class,
        pred.confidence AS confidence_score
    FROM (
        SELECT 
            image_path,
            predict_image_with_confidence_udf(image_path) AS pred
        FROM images
    ) AS sub
    """
)

print(f"query run time: {time.time() - start}")

# Show results
predictions_df.show()
predictions_df_confidence.show()

spark.stop()
