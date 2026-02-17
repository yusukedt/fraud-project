# src/evaluate.py

from pyspark.ml.evaluation import BinaryClassificationEvaluator
import json
from config import METRICS_PATH

def evaluate_model(predictions):

    roc_eval = BinaryClassificationEvaluator(
        labelCol="label",
        metricName="areaUnderROC"
    )

    pr_eval = BinaryClassificationEvaluator(
        labelCol="label",
        metricName="areaUnderPR"
    )

    metrics = {
        "AUC_ROC": roc_eval.evaluate(predictions),
        "AUC_PR": pr_eval.evaluate(predictions)
    }

    print("Metrics:", metrics)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics