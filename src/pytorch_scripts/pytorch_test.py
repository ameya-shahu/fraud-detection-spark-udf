from inference import ModelHandler

result = ModelHandler.predict(
    "E:\Files\ASU\sem_4\disml_final_project\spark-udf-cnn-inference\dataset\\all_images\generated.photos_v3_0008825.png"
)
print("Predicted class:", result)
