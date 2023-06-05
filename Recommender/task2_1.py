# packages
import os
import sys
import pyspark
import math
from itertools import combinations
import time
import random
import itertools
# Item Based Recommender with Pearson Correlation to decide the neighborhood
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

def score(neigh, user, user_dict, n):
    
    if not neigh:
        return 3 #default
    
    neigh = filter(lambda x: x[0] > 0, neigh)

    if not neigh:
        return 3 #default

    neigh = sorted(neigh, key=lambda x: x[0],reverse=True)
    
    num = min(len(neigh),n)
    
    cand = neigh[0:num]
    denom = 0
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


def write(result,file):
    f = open(file, "w")
    f.write("user_id, business_id, prediction\n")
    for p in result[1:]:
        f.write(",".join([str(i) for i in p]) + "\n")
    f.close()
   # with open(file, "w") as f:
        #f.write("user_id, business_id, prediction\n")
        #for i in result[1:]:
           #f.write(",".join([str(k) for k in i]) + "\n")
      
        #del f[1]
    #f.close()
    #with open(file, 'r') as f2:
        #lines = f2.readlines()
    #f2.close()
    #with open(file, 'w') as f3:
        #f3.writelines(lines[:1] + lines[2:])
    #f3.close()


if __name__ == '__main__':
    train_file_name = sys.argv[1]
    test_file_name = sys.argv[2]
    output_file_name = sys.argv[3]
    
    sc = pyspark.SparkContext('local[*]',appName="task2_1")
    sc.setLogLevel("WARN")
    start = time.time()
    train_rdd = sc.textFile(train_file_name)
    train = train_rdd.map(lambda x: x.strip().split(","))
    train_header = train.first()
    train = train.filter(lambda x: x != train_header).map(lambda x:(str(x[1]),str(x[0]),float(x[2])))
    
    test_rdd = sc.textFile(test_file_name)
    test = test_rdd.map(lambda x: x.strip().split(","))
    test_header = test_rdd.first()
    test = test.filter(lambda x: x!= test_header).map(lambda x:(str(x[1]),str(x[0])))
    
    user_train_dis = train.map(lambda x:x[1]).distinct().collect()
    user_test_dis = test.map(lambda x:x[1]).distinct().collect()
    user_dis = list(set(user_train_dis+user_test_dis))
    
    user_index_dict = {}
    j=0
    for u in user_dis:
        user_index_dict[u]=j
        j+=1
    # do the same for business
    bus_train_dis = train.map(lambda x:x[0]).distinct().collect()
    bus_test_dis = test.map(lambda x:x[0]).distinct().collect()
    bus_dis = list(set(bus_train_dis+bus_test_dis))
    
    bus_index_dict = {}
    k=0
    for b in bus_dis:
        bus_index_dict[b]=k
        k+=1
    
    index_user_dict = dict()
    for key,value in user_index_dict.items():
        index_user_dict[value]=key
 
    index_bus_dict = dict()
    for key,value in bus_index_dict.items():
        index_bus_dict[value]=key
    
    user_dict = train.map(lambda x:(user_index_dict[x[1]],(bus_index_dict[x[0]],x[2]))).groupByKey()
    user_dict = user_dict.map(lambda x:(x[0],list(x[1]))).collectAsMap()
    
    
    bus_dict = train.map(lambda x:(bus_index_dict[x[0]],(user_index_dict[x[1]],x[2]))).groupByKey()
    bus_dict = bus_dict.map(lambda x:(x[0],list(x[1]))).collectAsMap()
    
    for k,v in bus_index_dict.items():
        if v not in bus_dict.keys():
            bus_dict[v]=[]
        
    for k,v in user_index_dict.items():
        if v not in user_dict.keys():
            user_dict[v]=[]
        
    pred_temp = test.map(lambda x:(user_index_dict[x[1]],bus_index_dict[x[0]])).map(lambda x:(x[0],x[1],user_dict[x[0]])).map(lambda x:(x[0],x[1],[(pearson(x[1],bus_id,bus_dict),s) for (bus_id,s) in x[2]]))
    
    pred = pred_temp.map(lambda x: (index_user_dict[x[0]],index_bus_dict[x[1]],score(x[2],x[0],user_dict,2))).collect()
    end = time.time()
    
    write(pred,output_file_name)
    print("Duration:", end - start)