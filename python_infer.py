import os
import pandas as pd
import time
from pytorch_scripts.inference import ModelHandler

from conf import images_path_csv, pytorch_model_state_file, images_dir


# Initialize model
model_handler = ModelHandler(pytorch_model_state_file)

# Read CSV
df = pd.read_csv(images_path_csv)

# Add prediction columns
results = []

start = time.time()
for _, row in df.iterrows():
    image_path = os.path.join(images_dir, row["image_path"])

    if not os.path.isfile(image_path):
        print(f"Image not found: {image_path}")
        continue

    prediction = model_handler.predict(image_path)

    results.append(
        {
            "image_path": row["image_path"],
            "actual_class": row["actual_class"],
            "actual_class_id": row["actual_class_id"],
            "predicted_class": prediction["predicted_class"],
            "confidence": round(prediction["confidence"], 4),
            "class_0_prob": round(prediction["probabilities"][0], 4),
            "class_1_prob": round(prediction["probabilities"][1], 4),
            "class_2_prob": round(prediction["probabilities"][2], 4),
        }
    )


# Convert to DataFrame
results_df = pd.DataFrame(results)

print(f"Python inference time: {time.time() - start}")

# Optional: save to CSV
results_df.to_csv("dataset/inference_results.csv", index=False)


def compute_confusion_matrix(df: pd.DataFrame, positive_class: int = 1):
    """
    Compute TP, FP, TN, FN from a dataframe with columns:
    - 'actual_class_id'
    - 'predicted_class'

    :param df: DataFrame containing actual and predicted classes
    :param positive_class: The class to treat as the "positive" class (default=1)
    :return: confusion matrix DataFrame with TP, FP, TN, FN
    """
    tp = (
        (df["actual_class_id"] == positive_class)
        & (df["predicted_class"] == positive_class)
    ).sum()
    fp = (
        (df["actual_class_id"] != positive_class)
        & (df["predicted_class"] == positive_class)
    ).sum()
    tn = (
        (df["actual_class_id"] != positive_class)
        & (df["predicted_class"] != positive_class)
    ).sum()
    fn = (
        (df["actual_class_id"] == positive_class)
        & (df["predicted_class"] != positive_class)
    ).sum()

    cm_data = {
        "true_positives": [tp],
        "false_positives": [fp],
        "true_negatives": [tn],
        "false_negatives": [fn],
    }
    return pd.DataFrame(cm_data)


conf_matrix = compute_confusion_matrix(results_df, positive_class=1)

print(f"Python inference time with confusion matrix: {time.time() - start}")
