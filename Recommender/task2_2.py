# packages
import os
import sys
import pyspark
import math
from itertools import combinations
import time
import random
import itertools
import xgboost as xgb
import json
import numpy as np

def write(filename,rdd,prediction):
    with open (filename,'w')as f:
        f.write("user_id, business_id, prediction\n")
        temp = zip(rdd,prediction)
        for i in temp:
            f.write(i[0][0]+","+i[0][1]+"," +str(i[1])+"\n")
        f.close()


if __name__ == '__main__':
    folder_path = sys.argv[1]
    test_file_name = sys.argv[2]
    output_file_name = sys.argv[3]
    
    start = time.time()
    
    sc = pyspark.SparkContext(appName="task2_2")
    sc.setLogLevel("WARN")
    # ratings
    train_rate_file = folder_path+"/yelp_train.csv"
    train_rate_rdd = sc.textFile(train_rate_file)
    train_rate_rdd = train_rate_rdd.map(lambda x: x.strip().split(","))
    train_rate_header = train_rate_rdd.first()
    train_rate_rdd_filtered = train_rate_rdd.filter(lambda x: x!=train_rate_header)
    
    # collectAsMap() reference: https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.RDD.collectAsMap.html
    
    # metrics user features
    user_feature_file = folder_path+"/user.json"
    user_feature_rdd = sc.textFile(user_feature_file).map(lambda x: json.loads(x)).map(lambda x:(x["user_id"],[x["review_count"],x["average_stars"]])).collectAsMap()
    
    # metrics business
    bus_feature_file = folder_path+"/business.json"
    bus_feature_rdd = sc.textFile(bus_feature_file).map(lambda x:json.loads(x)).map(lambda x:(x["business_id"],[x["stars"],x["review_count"]])).collectAsMap()
    # training data
    X_train = np.array(train_rate_rdd_filtered.map(lambda x:np.array(user_feature_rdd[x[0]]+bus_feature_rdd[x[1]]).flatten()).collect())
    y_train = np.array(train_rate_rdd_filtered.map(lambda x: float(x[2])).collect())
    
    # testing data
    test_rdd = sc.textFile(test_file_name)
    test_rdd = test_rdd.map(lambda x: x.strip().split(","))
    test_rdd_header = test_rdd.first()
    test_rdd_filtered = test_rdd.filter(lambda x: x!=test_rdd_header)
    
    X_test = np.array(test_rdd_filtered.map(lambda x: np.array(user_feature_rdd[x[0]]+bus_feature_rdd[x[1]])).collect())
    
    # model
    parameter = {
        'random_state' : 30,
        'max_depth' : 15,
        'alpha' : 1,
        'booster':"gbtree",
        'n_estimators' : 200,
        'learning_rate' : 0.05
    }
    model = xgb.XGBRegressor(**parameter)
    model.fit(X_train,y_train)
    
    # prediction
    y_pred = model.predict(X_test)
    
    result = test_rdd_filtered.map(lambda x:(x[0],x[1])).collect()
    
    write(output_file_name, result,y_pred)
    end = time.time()
    
    print("Duration:",end-start)