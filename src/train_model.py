# src/train_model.py

from pyspark.ml.classification import LogisticRegression
from config import MODEL_PATH

from load_data import create_spark, load_dataset
from preprocess import prepare_features, split_data, add_class_weights
from evaluate import evaluate_model


def main():

    # 1. Spark
    spark = create_spark()

    # 2. Load
    df = load_dataset(spark)

    # 3. Preprocess
    data = prepare_features(df)
    train, test = split_data(data)
    train = add_class_weights(train)

    # 4. Train
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        weightCol="weight"
    )

    model = lr.fit(train)

    # 5. Predict
    predictions = model.transform(test)

    # 6. Evaluate
    evaluate_model(predictions)

    # 7. Save model
    model.write().overwrite().save(MODEL_PATH)

    print("Training complete. Model saved.")


if __name__ == "__main__":
    main()