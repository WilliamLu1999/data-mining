import sys
import json
import pyspark
import time
from operator import add

review_filepath = sys.argv[1]
output_filepath = sys.argv[2]
n_partition = int(sys.argv[3])
      
conf = pyspark.SparkConf().setAppName("task2").setMaster("local[*]")
sc = pyspark.SparkContext(conf = conf)
sc.setLogLevel("ERROR")
reviews_rdd = sc.textFile(review_filepath).map(lambda review: json.loads(review)).cache()
    
answer = dict()
default = dict()
customized = dict()


# default
start_default = time.time()

step1 = reviews_rdd.map(lambda x:[x["business_id"],1])

step2 = step1.reduceByKey(add).sortBy(lambda x:(-x[1],x[0])).take(10)

end_default = time.time()

default["n_partition"]=step1.getNumPartitions()
default["n_items"]=step1.glom().map(len).collect()
default["exe_time"]= end_default - start_default
# customized
start_customized = time.time()

step11 = reviews_rdd.map(lambda x:[x["business_id"],1]).partitionBy(n_partition,lambda x: ord(x[0][0]))

step22 = step11.reduceByKey(add).sortBy(lambda x:(-x[1],x[0])).take(10)

end_customized = time.time()


customized["n_partition"]=step11.getNumPartitions()
customized["n_items"] = step11.glom().map(lambda x:len(x)).collect()
customized["exe_time"]=end_customized-start_customized
answer["default"] = default
answer["customized"] = customized

sc.stop()

with open(output_filepath,'w') as f:
    json.dump(answer,f)