import sys
import json
import pyspark
import time
from operator import add

review_filepath = sys.argv[1]
business_filepath = sys.argv[2]
output_filepath_question_a = sys.argv[3]
output_filepath_question_b = sys.argv[4]


conf = pyspark.SparkConf().setAppName("task3").setMaster("local[*]")
sc = pyspark.SparkContext(conf = conf)
sc.setLogLevel("ERROR")

reviews_rdd = sc.textFile(review_filepath).map(lambda review: json.loads(review))
business_rdd = sc.textFile(business_filepath).map(lambda bi: json.loads(bi))
reviews_star = reviews_rdd.map(lambda review:(review["business_id"],review["stars"]))
business_city = business_rdd.map(lambda city: (city["business_id"],city["city"]))

# Average stars for every city
star_city = reviews_star.join(business_city)
temp = star_city.map(lambda rb:(rb[1][1],(rb[1][0],1))).reduceByKey(lambda x,y:(x[0]+y[0],x[1]+y[1])).map(lambda x: (x[0], float(x[1][0]/x[1][1])))

avg_star =temp.sortBy(lambda astar:(-astar[1],astar[0])).collect()


# compare execution time python and spark sorting

answer = {}
start = time.time()
m1 = temp.collect()
m1.sort(key = lambda x:(-x[1],x[0]))
print(m1[0:10])
end = time.time()
answer["m1"] = end-start

start2 = time.time()
m2 = temp.sortBy(lambda x:(-x[1],x[0])).take(10)
print(m2)
end2 = time.time()
answer["m2"]= end2-start2

answer["reason"] = "Sorting with python is within memory/RAM; whereas spark sorting has shuffling and many partitions, which might take longer."

with open(output_filepath_question_a,'w') as f:
    f.write("city,stars\n")
    for city, stars in avg_star:
        f.write(str(city)+","+str(stars)+"\n")

with open(output_filepath_question_b,'w') as ff:
    json.dump(answer,ff)