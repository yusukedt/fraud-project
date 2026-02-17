# src/preprocess.py

from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql.functions import when
from config import SEED, TRAIN_RATIO
from pyspark.sql.functions import sin, cos, col
import math

def prepare_features(df):

    df = df.withColumn("Time_sin", sin(col("Time") * 2 * math.pi / 86400))
    df = df.withColumn("Time_cos", cos(col("Time") * 2 * math.pi / 86400))
    df = df.drop("Time")

    # Scale Amount separately
    amount_assembler = VectorAssembler(inputCols=["Amount"], outputCol="amount_vec")
    df = amount_assembler.transform(df)

    scaler = StandardScaler(inputCol="amount_vec", outputCol="scaled_amount")
    scaler_model = scaler.fit(df)
    df = scaler_model.transform(df)

    # Drop original amount
    df = df.drop("Amount", "amount_vec")

    feature_cols = [c for c in df.columns if c not in ["Class"]]

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )

    data = assembler.transform(df).select("features", "Class")

    return data.withColumnRenamed("Class", "label")


def split_data(data):
    fraud = data.filter("label = 1")
    normal = data.filter("label = 0")

    train_fraud, test_fraud = fraud.randomSplit([TRAIN_RATIO, 1-TRAIN_RATIO], seed=SEED)
    train_normal, test_normal = normal.randomSplit([TRAIN_RATIO, 1-TRAIN_RATIO], seed=SEED)

    train = train_fraud.union(train_normal)
    test = test_fraud.union(test_normal)

    return train, test

def add_class_weights(train):
    fraud_count = train.filter("label = 1").count()
    normal_count = train.filter("label = 0").count()

    ratio = normal_count / fraud_count

    return train.withColumn(
        "weight",
        when(train.label == 1, ratio).otherwise(1.0)
    )