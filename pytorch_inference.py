import os
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType

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

# Start Spark Session
spark = SparkSession.builder.appName("ImagePrediction").getOrCreate()

# Register your Python function as a UDF returning String
spark.udf.register(
    "predict_image_udf",
    lambda path: predict_image(os.path.join(images_dir, path)),
    StringType()
)

# Load images DataFrame (assuming it has a column 'image_path')
image_df = spark.read.csv(images_path_csv, header=True, inferSchema=True)

# Create a temporary view to apply SQL
image_df.createOrReplaceTempView("images")

# Use Spark SQL to call the UDF
predictions_df = spark.sql("""
    SELECT
        image_path,
        predict_image_udf(image_path) AS predicted_class
    FROM images
""")

# Show results
predictions_df.show()

spark.stop()
