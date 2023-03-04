import os
import sys
import pyspark
import math
from itertools import combinations
import time
import itertools

# 

def get_frequent_itemset(items, bar):

    clean_frequent_items = []
    for i, c in items.items():
        if c >= bar:
            clean_frequent_items.append(i)
    return clean_frequent_items 


def frequent_candidate_generator(lines, temp, bar):
    result = dict()
    for line in lines:
        if bar == 1:
            for l in line:
                l = (l,)
                if l not in result:
                    result[l]=1
                else:
                    result[l]+=1
                
        else:
            inter = sorted(list(set(line).intersection(temp)))
            
            for l in combinations(inter, bar):
                if l not in result:
                    result[l]=1
                else:
                    result[l]+=1
              
    return result

def SON_candidate(parts, supt, bar):
    result = []
    parts = list(parts)
    parts_len = len(parts)
    threshold = math.ceil(parts_len * supt/bar)

    n = 1
    candidates = frequent_candidate_generator(parts, [], n)
    candidates_filtered = get_frequent_itemset(candidates, threshold)
    
    if candidates_filtered:
        result.append(candidates_filtered)
        
    if len(candidates_filtered) <= 1:
        return result
    
    n += 1
    candidates_filtered = set(map(lambda x: x[0], candidates_filtered))
    

    
    while candidates_filtered:
        candidates = frequent_candidate_generator(parts, candidates_filtered, n)
        candidates_filtered = get_frequent_itemset(candidates, threshold)

        if candidates_filtered:
            result.append(candidates_filtered)
            
        if len(candidates_filtered) <= 1:
            break
        
        temp_list =[]
        for can in candidates_filtered:
            for c in can:
                temp_list.append(c)
        temp_list= set(temp_list)    
           
        candidates_filtered =temp_list  
        n += 1

    return result


def SON_frequent(part, candidates):
    result = dict()
    for p in part:
        for candidate in candidates:
            if set(candidate).issubset(set(p)):
                if candidate not in result:
                    result[candidate]=1
                else:
                    result[candidate]+=1
                
    return [(x, y) for x, y in result.items()]
        

def export_output(data):
    temp = ''
    temp_len = 1
    for d in data:
        if len(d) == 1:
            temp += f'{str(d).split(",")[0]}),'
        elif len(d) == temp_len:
            temp += f'{d},'
        else:
            temp = temp[:-1]
            temp += f'\n\n{d},'
            temp_len = len(d)
    
    return temp.rstrip(",")

case_number = int(sys.argv[1])
support = int(sys.argv[2])
input_file_path = sys.argv[3]
output_file_path = sys.argv[4]
sc = pyspark.SparkContext.getOrCreate()
sc.setLogLevel("ERROR")
start = time.time()

rdd = sc.textFile(input_file_path).map(lambda x:x.strip().split(","))

if case_number == 1:
    bkt = rdd.filter(lambda x: x!='user_id').map(lambda y:(str(y[0]),str(y[1])))
    basket = bkt.groupByKey().mapValues(list).map(lambda x: sorted(list(set(x[1]))))
else:
    bkt = rdd.filter(lambda x: x!='user_id').map(lambda y:(str(y[1]),str(y[0])))
    basket = bkt.groupByKey().mapValues(list).map(lambda x: sorted(list(set(x[1]))))

basket_len = basket.count()

# find candidates
candidates = basket.mapPartitions(lambda x: SON_candidate(x,support,basket_len)).flatMap(lambda x:x).distinct().sortBy(lambda x:(len(x),x)).collect()


frequent_items = basket.mapPartitions(lambda x: SON_frequent(x,candidates)).reduceByKey(lambda x,y: x+y).filter(lambda x:x[1]>=support).map(lambda x: x[0])
frequent_items = frequent_items.sortBy(lambda x:(len(x),x)).collect()                                

end = time.time()

#jjj = ''
print("Duration:",end-start)                              
#print(candidates)  
#print(frequent_items)
#print(candidates[0])
#print(len(candidates[0]))
#print(len(candidates[8]))
#jjj += f"{str(candidates[0]).split(',')[0]}),"
#print(jjj)

with open(output_file_path, 'w+') as f:
    f.write('Candidates:\n' + export_output(candidates) + '\n\n' +'Frequent Itemsets:\n' + export_output(frequent_items))