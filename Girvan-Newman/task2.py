import sys
import time
import os
import sys
import pyspark
import math
from itertools import combinations
from itertools import permutations
from collections import defaultdict
import copy
import time
import random
import itertools
import json
import numpy as np


def write_betweeness(path,result):
    with open(path, "w") as f:
        for u, v in result:
            u2 = str(u)
            output = u2 + "," + str(round(v, 5)) + "\n" 
            f.write(output)   

def write_community(path,result):
    with open(path,"w") as f:
        for u in result:
            f.write(str(u)[1:-1] + "\n")
            
def GirvanNewman(input_, nodes):
    between = defaultdict(float)
    for root in nodes:
        path = []
        tree = []
        level = dict()
        node_weight = dict()
        parent = defaultdict(set)
        shortest_path = defaultdict(float) # BFS
        level[root] = 0
        shortest_path[root] = 1
        edge_weight = defaultdict(float)
        adjacent = set()
        tree.append(root)
        adjacent.add(root)
        while len(tree) > 0:
            root = tree.pop(0)
            path.append(root)
            
            for x in input_[root]:
                if x not in adjacent:
                    tree.append(x)
                    adjacent.add(x)
                    
                    if x in parent.keys():
                        parent[x].add(root)
                        
                    else:
                        parent[x] = set()
                        parent[x].add(root)
                        
                    shortest_path[x] += shortest_path[root]
                    level[x] = level[root] + 1
                    
                elif level[x] == level[root] + 1:
                    if x in parent.keys():
                        parent[x].add(root)
                        
                    else:
                        parent[x] = set()
                        parent[x].add(root)
                        
                    shortest_path[x] += shortest_path[root]
        
        for x in path:
            node_weight[x] = 1
            
        reverse_path = reversed(path)
        
        for x in reverse_path:
            for i in parent[x]:
                temp_weight = shortest_path[i] / shortest_path[x]
                temp_weight = temp_weight*node_weight[x]
                node_weight[i] += temp_weight
                temp_key= sorted([x, i])
                temp_key_tup = tuple(temp_key)
                
                if temp_key_tup in edge_weight.keys():
                    edge_weight[temp_key_tup] += temp_weight
                else:
                    edge_weight[temp_key_tup] = temp_weight
                    
        for k, v in edge_weight.items():
            if k in between.keys():
                between[k] += v / 2
            else:
                between[k] = v / 2
                
    between = sorted(between.items(), key=lambda x: (-x[1], x[0]))

    return between
            
def create_dic(u_, edges):
    
    for u1, u2 in edges:
        if u1 not in u_:
            u_[u1] = set()
        if u2 not in u_:
            u_[u2] = set()
            
        u_[u1].add(u2)
        u_[u2].add(u1)
    
    return u_    

                    
if __name__ == '__main__':
    # inputs
    filter_threshold = int(sys.argv[1])
    input_file_path = sys.argv[2]
    betweeness_output_file_path = sys.argv[3]
    community_output_file_path = sys.argv[4]
    
    start = time.time()
    
    sc = pyspark.SparkContext(appName="task2")
    sc.setLogLevel("WARN")
    rdd = sc.textFile(input_file_path)
    header = rdd.first()
 
    rdd2 = rdd.filter(lambda x: x!= header).map(lambda x:x.strip().split(","))
    users = rdd2.map(lambda x:x[0]).distinct() 
    users2 = users.collect()
    user_business = rdd2.groupByKey().mapValues(set)
    
    user_business_dict = dict()
    for u, b in user_business.collect():
        user_business_dict[u] = b
    

    edges = []
    nodes = set()
    for u in list(permutations(users2,2)):
        temp0 = u[0]
        temp1 = u[1]
        if len(user_business_dict[temp0] & user_business_dict[temp1]) >= filter_threshold:
            edges.append(u)
            nodes.add(temp0)
    nodess = list(nodes)
    
    # calculate betweeness
    u_1 = dict()
    u_ = create_dic(u_1,edges)
    betweeness = GirvanNewman(u_, nodess)
    
    write_betweeness(betweeness_output_file_path,betweeness) 
    
    #######################
    u_2 = copy.deepcopy(u_)
    m = len(betweeness)
    mod = -math.inf
    
    k = dict()
    for u in u_:
        k[u]=len(u_[u])


    # betweeness_length = len(betweeness)
    while len(betweeness) > 0:
        nodess2 = nodess.copy()
        members = []
     
        while len(nodess2) > 0:
            root = nodess2.pop()
            tree = []
            tree.append(root)
            adjacent = set()
            adjacent.add(root)
            
            while len(tree) > 0:
                root = tree.pop(0)
                for n in u_2[root]:
                    if n not in adjacent:
                        tree.append(n)
                        adjacent.add(n)
                        nodess2.remove(n)
                        
            adjacent_ = sorted(list(adjacent))
            members.append(adjacent_)

        modularity = float(0)
        for member in members:
            for a in member:
                for b in member:
                    if b not in u_[a]:
                        A = float(0)
                    else:
                        A = float(1)
                   
                    
                    modularity += A - (k[a] * k[b]) / (2.0 * m)
                    
        modularity /= (2 * m)

        if mod<modularity:
            mod = modularity
            community = copy.deepcopy(members)

        betweenness_max = betweeness[0][1]
        for v, b in betweeness:
            v0 = v[0]
            v1 = v[1]
            if b>=betweenness_max:
                u_2[v0].remove(v1)
                u_2[v1].remove(v0)
                
        betweeness = GirvanNewman(u_2, nodess)

    community = sorted(community, key=lambda x: (len(x), x[0]))
    
    write_community(community_output_file_path,community)
    
    end = time.time()
    print('Duration:', end - start)
    print("success!")