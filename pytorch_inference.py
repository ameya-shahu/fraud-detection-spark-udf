import os
import time
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType
from pyspark.sql.functions import rand
from conf import pytorch_model_state_file, images_path_csv, images_dir, model_lable_map, predication_detail_query, confusion_matrix_query


def predict_image(image_path):
    """
    Lazy load the PyTorch model once in each executor (worker).
    Then run inference and map the numeric class to a string label.
    """
    if not hasattr(predict_image, "model_handler"):
        from pytorch_scripts.inference import ModelHandler

        print(f"Loading model from: {pytorch_model_state_file}")
        predict_image.model_handler = ModelHandler(pytorch_model_state_file)

    
    pred_details = predict_image.model_handler.predict(image_path)


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
    "predict_image",
    lambda path: predict_image(os.path.join(images_dir, path)),
    schema,
)

# Load images DataFrame (assuming it has a column 'image_path')
image_df = spark.read.csv(images_path_csv, header=True, inferSchema=True)

# Create a temporary view to apply SQL
image_df.createOrReplaceTempView("images")

start = time.time()
predictions_df = spark.sql(predication_detail_query)
print(f"Prediction Details Query Time: {time.time() - start}")

start = time.time()
confusion_ma_df = spark.sql(confusion_matrix_query)
print(f"Confusion Matrix Query Time: {time.time() - start}")

# Show results
predictions_df.orderBy(rand()).show(truncate=False, n=30)
predictions_df.coalesce(1).write.mode("overwrite").option("header", True).csv("output_predictions")

confusion_ma_df.show()

spark.stop()
