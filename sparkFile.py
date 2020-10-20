import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

# Entry gate - provides access to resource manager (YARN (or MESOS))
sc = SparkContext()

# Spark Session
spark = SparkSession.builder \
    .master('local') \
    .appName('PySpark Tutorial') \
    .getOrCreate()

# Read data
df = spark.read.format('csv') \
    .option('header', True) \
    .option('multiLine', True) \
    .load('data/data.csv')
df.show()
print(f'Record count is: {df.count()}')





