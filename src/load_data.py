# src/load_data.py

from pyspark.sql import SparkSession
from config import DATA_PATH

def create_spark():
    return SparkSession.builder \
        .appName("CreditCardFraudDetection") \
        .getOrCreate()

def load_dataset(spark):
    df = spark.read.csv(
        DATA_PATH,
        header=True,
        inferSchema=True
    )
    return df