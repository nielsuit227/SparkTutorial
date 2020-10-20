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
# Train / Test Split
splits = vectorDf.randomSplit([0.7, 0.3])
trainDf = splits[0]
testDf = splits[1]
# Linear Regression
lr = LinearRegression(featuresCol = 'features', labelCol='MV', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(trainDf)
# Outputs
trainingSummary = lr_model.summary
test_result = lr_model.evaluate(testDf)
print("Coefficients: " + str(lr_model.coefficients))
print("R2-score on train data: %f" % trainingSummary.r2)
print("R2-score on test data = %f" % test_result.r2)