import numpy as np
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

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
# Transform to Floats
for col in df.columns:
    df = df.withColumn(col, df[col].cast('float').alias(col))
# Checking data
print(f'Record count is: {df.count()}')
df.show()
df.printSchema()
df.describe().toPandas().transpose()
# Split input/output
splitter = VectorAssembler(inputCols=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',
                                      'B', 'LSTAT'], outputCol = 'features')
vectorDf = splitter.transform(df)
vectorDf = vectorDf.select(['features', 'MV'])
vectorDf.show(3)
# Linear Regression