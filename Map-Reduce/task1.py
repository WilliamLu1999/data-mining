import sys
import json
import re
import pyspark
from operator import add



review_filepath = sys.argv[1]
output_filepath = sys.argv[2]
conf = pyspark.SparkConf().setAppName("task1").setMaster("local[*]")
sc = pyspark.SparkContext(conf = conf)
sc.setLogLevel("ERROR")
reviews_rdd = sc.textFile(review_filepath).map(lambda review: json.loads(review)).cache()

# answers need to be in JSON format so
answer = {}
# A
answer['n_review'] = reviews_rdd.count()
# B
answer['n_review_2018'] = reviews_rdd.map(lambda review: review['date']).filter(lambda date:'2018' in date).count()
# C
answer["n_user"] = reviews_rdd.map(lambda review: review["user_id"]).distinct().count()
# D
answer['top10_user']= reviews_rdd.map(lambda review: (review['user_id'],1)).reduceByKey(add).sortBy(lambda x:(-x[1],x[0])).take(10)
# E
answer['n_business']= reviews_rdd.map(lambda review: review["business_id"]).distinct().count()
# F 
answer['top10_business']=reviews_rdd.map(lambda review: (review['business_id'],1)).reduceByKey(add).sortBy(lambda x:(-x[1],x[0])).take(10)
#print("hi")
sc.stop()
with open(output_filepath, 'w') as f:
    json.dump(answer, f)

    