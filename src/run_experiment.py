"""Replikasi parsial alur Spark SQL + MLlib dari Ali et al. (2024)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import PolynomialExpansion, VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor, LinearRegression
from pyspark.sql import SparkSession, functions as F

SEED = 42
RAW_PATH = "data/raw/lcl_sample.csv"
RESULTS = Path("results")


def metric_rows(name, predictions, elapsed_seconds):
    evaluators = {
        "RMSE": RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse"),
        "MAE": RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae"),
        "R2": RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2"),
    }
    return [{"model": name, "metric": metric, "value": evaluator.evaluate(predictions),
             "fit_seconds": elapsed_seconds} for metric, evaluator in evaluators.items()]


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    spark = (SparkSession.builder.appName("LCLPartialReplication")
             .master("local[*]").config("spark.sql.shuffle.partitions", "8").getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    try:
        raw = spark.read.option("header", True).option("inferSchema", True).csv(RAW_PATH)
        cleaned = (raw.withColumn("timestamp", F.to_timestamp("DateTime"))
                   # Nama kolom ini berasal persis dari CSV UK Power Networks.
                   .withColumn("kwh", F.trim(F.col("KWH/hh (per half hour) ")).cast("double"))
                   .filter(F.col("timestamp").isNotNull() & F.col("kwh").isNotNull() & (F.col("kwh") > 0)))

        # Artikel melakukan analitik deskriptif sebelum feature selection. Di sini
        # pembacaan tiap rumah diagregasi per interval 30 menit sebagai target beban.
        series = (cleaned.groupBy("timestamp").agg(F.sum("kwh").alias("label"),
                                                     F.countDistinct("LCLid").alias("meter_count"))
                  .withColumn("hour", F.hour("timestamp").cast("double"))
                  .withColumn("day_of_week", F.dayofweek("timestamp").cast("double"))
                  .withColumn("month", F.month("timestamp").cast("double"))
                  .withColumn("is_weekend", F.when(F.dayofweek("timestamp").isin(1, 7), 1.0).otherwise(0.0))
                  .orderBy("timestamp").cache())
        n_rows = series.count()
        if n_rows < 100:
            raise RuntimeError("Data setelah pembersihan terlalu sedikit untuk pelatihan.")
        cutoff_epoch = series.select(F.col("timestamp").cast("long").alias("epoch")).approxQuantile("epoch", [0.8], 0.0)[0]
        cutoff = spark.sql(f"SELECT timestamp_seconds({int(cutoff_epoch)}) AS cutoff").first()["cutoff"]
        train, test = series.filter(F.col("timestamp") <= F.lit(cutoff)), series.filter(F.col("timestamp") > F.lit(cutoff))
        feature_cols = ["hour", "day_of_week", "month", "is_weekend", "meter_count"]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")

        polynomial = Pipeline(stages=[assembler, PolynomialExpansion(inputCol="features_raw", outputCol="features", degree=2),
                                      LinearRegression(featuresCol="features", labelCol="label", maxIter=100, regParam=0.01)])
        t0 = time.perf_counter(); poly_model = polynomial.fit(train); poly_elapsed = time.perf_counter() - t0
        poly_predictions = poly_model.transform(test).select("timestamp", "label", "prediction")

        tree = Pipeline(stages=[assembler, DecisionTreeRegressor(featuresCol="features_raw", labelCol="label", maxDepth=8, seed=SEED)])
        t0 = time.perf_counter(); tree_model = tree.fit(train); tree_elapsed = time.perf_counter() - t0
        tree_predictions = tree_model.transform(test).select("timestamp", "label", "prediction")

        metrics = metric_rows("Regresi polinomial derajat 2", poly_predictions, poly_elapsed)
        metrics += metric_rows("Decision tree depth 8", tree_predictions, tree_elapsed)
        pd.DataFrame(metrics).to_csv(RESULTS / "metrics.csv", index=False)

        # Simpan prediksi tree untuk audit serta visualisasi 7 hari pertama dari data uji.
        preview = tree_predictions.orderBy("timestamp").limit(336).toPandas()
        preview.to_csv(RESULTS / "predictions.csv", index=False)
        plt.figure(figsize=(11, 4.5))
        plt.plot(preview["timestamp"], preview["label"], label="Aktual", linewidth=1.2)
        plt.plot(preview["timestamp"], preview["prediction"], label="Prediksi decision tree", linewidth=1.0)
        plt.xlabel("Waktu"); plt.ylabel("Total kWh per 30 menit"); plt.legend(); plt.tight_layout()
        plt.savefig(RESULTS / "actual_vs_prediction.png", dpi=180); plt.close()
        metadata = {"seed": SEED, "raw_records_after_cleaning": cleaned.count(), "aggregate_rows": n_rows,
                    "train_rows": train.count(), "test_rows": test.count(), "cutoff_timestamp": str(cutoff),
                    "spark_version": spark.version, "features": feature_cols}
        (RESULTS / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(pd.DataFrame(metrics).to_string(index=False))
        print(json.dumps(metadata, indent=2))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
