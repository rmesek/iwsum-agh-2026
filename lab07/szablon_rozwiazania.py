from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.ml.feature import *


spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

# tu jest miejsce na twoje rozwiazanie

spark.stop()
