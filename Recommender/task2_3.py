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

def pearson(i1, i2, d):
    i1_scores = d[i1]
    i2_scores = d[i2]

    if not i1_scores or not i2_scores:
        return 0
    
    i1_dict = dict()
    for (key, value) in i1_scores:
        i1_dict[key] = value
    i2_dict = dict()
    for (key, value) in i2_scores:
        i2_dict[key] = value
    
    i1_sums = sum(i1_dict.values())
    i1_avg = i1_sums / len(i1_dict)
    # do the same
    i2_sums = sum(i2_dict.values())
    i2_avg = i2_sums / len(i2_dict)
    
    nomi = 0
    denom_i1 = 0
    denom_i2 = 0
    
    corated = set(i1_dict.keys())& set(i2_dict.keys())
    
    
    if len(corated) < 70:
        return 0

    for item in corated:
      
        nomi += (i1_dict[item] - i1_avg)*(i2_dict[item] - i2_avg)
        denom_i1 += (i1_dict[item] - i1_avg) ** 2
        denom_i2 += (i2_dict[item] - i2_avg) ** 2

    if (denom_i1*denom_i2) == 0:  #
        return 0
    
    denom = math.sqrt(denom_i1) * math.sqrt(denom_i2)
    return nomi / denom

def create_dict(dis,dic):
        temp = 0
        for d in dis:
            dic[d]= temp
            temp+=1
        return dic
    
    
def score(neigh, user, user_dict, n):
    
    
    if not neigh:
        return 3 #default
    
    neigh = filter(lambda x: x[0] > 0, neigh)

    if not neigh:
        return 3 #default

    neigh = sorted(neigh, key=lambda r: -r[0])
    
    num = min(len(neigh),n)

    denom = 0
    cand  = neigh[0:num]
    for (a,b) in cand:
        denom+=abs(a)
        
    if denom == 0:
        user_rating = user_dict[user]
        if not user_rating:
            return 3
        else:
            temp = 0
            for (a,b) in user_rating:
                temp += b
            return temp/len(user_rating)
    else:
        numer = 0
        for (a,b) in cand:
            numer +=a*b

        return numer/denom
    
def default_value(val):
    if np.isnan(val) or not val or val==float("inf") or val ==float("-inf"):
        return 3
    else:
        return val
    
    
    
def write(result,file):
    with open(file, "w") as f:
        f.write("user_id, business_id, prediction\n")
        for i in result:
            f.write(",".join([str(k) for k in i]) + "\n")
       
    f.close()
    
    
    
if __name__ == '__main__':
    folder_path = sys.argv[1]
    test_file_name = sys.argv[2]
    output_file_name = sys.argv[3]
    
    start = time.time()


    sc = pyspark.SparkContext(appName="task2_3")
    sc.setLogLevel("WARN")
    
    # model based
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
    X_train = np.array(train_rate_rdd_filtered.map(lambda x: np.array(user_feature_rdd[x[0]]+bus_feature_rdd[x[1]])).collect())
    y_train = np.array(train_rate_rdd_filtered.map(lambda x: float(x[2])).collect())
    
    # testing data
    test_rdd = sc.textFile(test_file_name)
    test_rdd = test_rdd.map(lambda x: x.strip().split(","))
    test_rdd_header = test_rdd.first()
    test_rdd_filtered = test_rdd.filter(lambda x: x!=test_rdd_header)
    
    X_test = np.array(test_rdd_filtered.map(lambda x: np.array(user_feature_rdd[x[0]]+bus_feature_rdd[x[1]])).collect())
    
    # model
    parameter = {
        'random_state' : 5,
        'max_depth' : 10,
        'alpha' : 1,
        'booster':"gbtree",
        'n_estimators' : 100,
        'learning_rate' : 0.1
    }
    model = xgb.XGBRegressor(**parameter)
    model.fit(X_train,y_train)
    
     # prediction
    y_pred = model.predict(X_test)
    test_tuple = test_rdd_filtered.map(lambda x: (x[0], x[1])).collect()
  
    y_pred2 = []
    for val in y_pred:
        cleaned = default_value(val)
        y_pred2.append(cleaned)
        
    # CF item-based
    # item-based
    train_rdd = train_rate_rdd_filtered.map(lambda x: (str(x[1]), str(x[0]), float(x[2])))
    test_rdd2 = test_rdd_filtered.map(lambda x: (str(x[1]), str(x[0])))
    
    
    # distinct users collection
    
    user_train_dis = train_rdd.map(lambda x: x[1]).distinct().collect()
    user_test_dis = test_rdd2.map(lambda x: x[1]).distinct().collect()
    user_dis = list(set(user_train_dis + user_test_dis))
    
    
    
    bus_train_dis = train_rdd.map(lambda x: x[0]).distinct().collect()
    bus_test_dis = test_rdd2.map(lambda x: x[0]).distinct().collect()
    bus_dis = list(set(bus_train_dis + bus_test_dis))
    
    user_index_dict = dict()
    user_index_dict = create_dict(user_dis,user_index_dict)
    
    bus_index_dict = dict()
    bus_index_dict = create_dict(bus_dis,bus_index_dict)
     
    index_user = dict()
    for key,value in user_index_dict.items():
        index_user[value]=key
    
    index_bus = dict()
    for key,value in bus_index_dict.items():
        index_bus[value]=key
    

    user_dict = train_rdd.map(lambda x: (user_index_dict[x[1]], (bus_index_dict[x[0]], x[2]))).groupByKey()
    user_dict = user_dict.map(lambda x: (x[0], list(x[1]))).collectAsMap()

  

    bus_dict = train_rdd.map(lambda x: (bus_index_dict[x[0]], (user_index_dict[x[1]], x[2]))).groupByKey()
    
    bus_dict = bus_dict.map(lambda x: (x[0], list(x[1]))).collectAsMap()
        
    for k, v in bus_index_dict.items():
        if v not in bus_dict.keys():
            bus_dict[v]=[]
        else:
            continue
    for k, v in user_index_dict.items():
        if v not in user_dict.keys():
            user_dict[v]=[]
        else:
            continue

    pred = test_rdd2.map(lambda x: (user_index_dict[x[1]], bus_index_dict[x[0]])).map(lambda x: (x[0], x[1], user_dict[x[0]]))
    
    pred = pred.map(lambda x: (
    x[0], x[1], [(pearson(x[1], bus_id, bus_dict), score) for (bus_id, score) in x[2]])).map(lambda x: (x[0], x[1], score(x[2], x[0], user_dict, 2)))
    
    
    result = dict()
    for i in range(len(y_pred2)):
        user, bus = test_tuple[i]
        result[(user_index_dict[user],bus_index_dict[bus])]=y_pred2[i]
        
    # collaborative filtering    
    alpha = 0.11
    pred2 = pred.map(lambda x: (x[0], x[1], x[2] * alpha + result[(x[0], x[1])] * (1-alpha))).map(lambda x: (index_user[x[0]], index_bus[x[1]], x[2])).collect()
    
        
    end = time.time()
    write(pred2,output_file_name)
    print("Duration:", end - start)
    
                    
                    