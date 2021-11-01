import pyspark
from pyspark import SparkContext,SparkConf
from pyspark.sql.session import SparkSession
from pyspark.sql.types import StructType,StructField,StringType
from pyspark.sql.functions import col
from pyspark.sql.functions import concat,isnull,when,count,col

directory="./jars/*"


spark = SparkSession \
    .builder \
    .appName("NewsClassifier_Datapreparation") \
    .config("spark.mongodb.input.uri", "mongodb://localhost:27017/week2.news") \
    .config("spark.mongodb.output.uri", "mongodb://localhost:27017/week2.news") \
    .config('spark.driver.extraClassPath', directory) \
    .getOrCreate()


print(spark)

df = spark.read.format("mongo").load()

print("Schema:")
df.printSchema()

print("show top 5 records: ")
df.show(5)

print("Number of records: ",df.count())
print("Columns: ", df.columns)

#model prediction
#columns selection for knowing the news details
dataset = df.select(col('category'), col('headline'),  col('short_description') )
dataset.show()

#data cache -loading it into local dataset
#dataset.cache()s
dataset.fillna('.')

#cleaning the data
dataset.select([when(isnull(c),c).alias(c) for c in dataset.columns]).show()
dataset=dataset.replace('?',None).dropna(how='any')



