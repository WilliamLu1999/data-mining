# packages

import os
import sys
import pyspark
from pyspark.sql import SQLContext
import math
from itertools import combinations
import time
import random
import itertools
import xgboost as xgb
import json
import numpy as np
import graphframes


def write(output_file_path,df):
    with open(output_file_path,'w') as f:
        for i in df.collect():
            f.write(str(i)[1:-1] + "\n")
    f.close()

if __name__ == '__main__':
    filter_threshold = int(sys.argv[1])
    input_file_path = sys.argv[2]
    community_output_file_path = sys.argv[3]
    #spark-submit --packages graphframes:graphframes:0.8.2-spark3.1-s_2.12 task1.py <filter threshold> <input_file_path> <community_output_file_path>
    start = time.time()
    
    sc = pyspark.SparkContext(appName="task1")
    sc.setLogLevel("WARN")
    sqlContext = SQLContext(sc) # create a sql context
    
    rdd = sc.textFile(input_file_path)
    header = rdd.first()
    
    rdd2 = rdd.filter(lambda x: x!= header).map(lambda x:x.strip().split(","))
    users = rdd2.map(lambda x:x[0]).distinct() # distinct users
    users2 = users.collect()
    businesses = rdd2.map(lambda x:x[1]).distinct() # distinct businesses
    user_bus_dict = rdd2.map(lambda x:(x[0],x[1])).groupByKey()
    
    user_bus_dict = user_bus_dict.map(lambda x:(x[0],sorted(list(x[1])))).mapValues(set).collectAsMap()
    
 
            
    edges = []
    nodes = set()
    for p in list(combinations(users2, 2)):
        temp1 = p[0]
        temp2 = p[1]
        if len((user_bus_dict[temp1]).intersection(user_bus_dict[temp2])) >= filter_threshold:
            edges.append((temp1, temp2))
            edges.append((temp2, temp1))
            nodes.add(temp1)
            nodes.add(temp2)
            
    edges = list(edges)
    
    nodes = list(nodes)
    
    
    nodes_rows = []
    for v in nodes:
        row = (v,)
        nodes_rows.append(row)


    nodes_df = sqlContext.createDataFrame(nodes_rows, ["id"])
    edges_df = sqlContext.createDataFrame(edges).toDF("src", "dst")
    
    
    graph = graphframes.GraphFrame(nodes_df, edges_df)
    result = graph.labelPropagation(maxIter=5)
    
    groups_rdd = result.rdd.map(lambda x: (x[1], x[0])).groupByKey().map(lambda x: sorted(list(x[1])))
    groups_rdd = groups_rdd.sortBy(lambda x:(x[0])).sortBy(lambda x: (len(x)))
    
    write(community_output_file_path,groups_rdd)


    end = time.time()
    print("Duration:",end-start)