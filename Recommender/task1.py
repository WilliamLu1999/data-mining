# packages
import os
import sys
import pyspark
import math
from itertools import combinations
import time
import random
import itertools
# implement Locality Sensitive Hashing with Jaccard Similarity

def hash_coef(n1,m1):
    hash_lst = []
    a = random.sample(range(1, m1), n1)
    b = random.sample(range(1, m1), n1)
    hash_lst.append([a, b])
    return hash_lst

    
def jaccard_similarity(cand,b_u):
    out = {}
    for i, j in cand:
        temp1 = b_u[i]
        temp2 = b_u[j]
        similarity = len(temp1&temp2)/len(temp1|temp2)
        if similarity >=0.5:
            output[str(i)+","+str(j)]=similarity
    out = dict(sorted(output.items(),key=lambda x: x[0]))
    return out

def get_user_index_dict(rd):
    user_index_dict = {}
    j=0
    for i,u in enumerate(rd):
        user_index_dict[u]=i
        j+=1
    return user_index_dict,j

def write(dic):
    cont = "business_id_1,business_id_2,similarity\n"
    for k,v in dic.items():
        cont += str(k)+","+str(v)+"\n"
    return cont

if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    sc = pyspark.SparkContext(appName="task1")
    
    start = time.time()
    hash_num = 120
    row = 2
    band = hash_num // row # here is 60
    lines = sc.textFile(input_path).map(lambda x: x.strip().split(","))
    header = lines.first() 
    lines = lines.filter(lambda x: x != header)
    rdd = lines.map(lambda x: (x[1], x[0])).groupByKey().mapValues(set).collect()
    # business user
    bus_users = dict()
    for bus, user in rdd:
        bus_users[bus]=user
    # get distinct users
    distinct_users = lines.map(lambda x:x[0]).distinct().collect()
    user_index,lg = get_user_index_dict(distinct_users)
    #user_len = len(user_index)
    hashed_user_dict = hash_coef(hash_num,lg)
    hashed_user_dict = list(itertools.chain(*hashed_user_dict))
    # get signature matrix
    p = 90007
    m = lg
    sig ={}
    for a, b in rdd:
        minhash_lst = []
        for i in range(hash_num):
            for user in b:
                minhash = ((hashed_user_dict[0][i]* user_index[user]+hashed_user_dict[1][i])%p)%m
            minhash_lst.append(int(minhash))
        sig[a] = minhash_lst   
    
    # b bands and r rows each
    band_row_dict =dict()
    band_row_final =dict()
    for k, ms in sig.items():
        for i in range(0,band):
            br = tuple(ms[i*row:(i+1)*row])
            temp = (i,br)
            if temp in band_row_dict.keys():
                band_row_dict[temp].append(k)
            else:
                band_row_dict[temp] =[k]
    # iterate thru the dict to keep the final 
    for k, v in band_row_dict.items():
        val_len= len(v)
        if val_len <=1:
            continue
        else:
            band_row_final[k]=v
        
    # generate candidates
    candidates = set()
    for k, v in band_row_final.items():
        temp2 = combinations(sorted(v),row)
        for t in temp2:
            candidates.add(t)
    # get output
    output= dict()
    output = jaccard_similarity(candidates,bus_users)
    content = write(output)
    
    with open(output_path,"w") as f:
        f.write(content)
    end =time.time()
    duration = end-start
    print(f"Duraction:{duration}")