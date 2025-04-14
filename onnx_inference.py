import os
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType

# Import your configs
from conf import onnx_model_state_file, images_path_csv, images_dir, model_lable_map 


def predict_image(image_path):
    """
    Lazy load the model once per worker,
    then run inference on image_path, 
    and return a string label.
    """
    if not hasattr(predict_image, "model"):
        from onnx_scripts.inference import ONNXModel
        predict_image.model = ONNXModel(onnx_model_state_file)

    # Predict returns a numeric class (likely 0,1,2). Convert to int in case it's NumPy type.
    numeric_pred = int(predict_image.model.predict(image_path=image_path))

    # Return the mapped label (default to "Unknown" if out of range)
    return model_lable_map.get(numeric_pred, "Unknown")

# Initialize Spark
spark = SparkSession.builder.appName("ImagePrediction").getOrCreate()

# Register our Python function as a Spark SQL UDF returning String
spark.udf.register(
    "predict_image_udf",
    lambda p: predict_image(os.path.join(images_dir, p)),
    StringType()
)

# Read your CSV into a DataFrame
image_df = spark.read.csv(images_path_csv, header=True, inferSchema=True)

# Create a temporary view so we can query it
image_df.createOrReplaceTempView("images")

# Use Spark SQL to select predictions
predictions_df = spark.sql("""
    SELECT
        image_path,
        predict_image_udf(image_path) AS predicted_type
    FROM images
""")

# Show the results
predictions_df.show()

spark.stop()
