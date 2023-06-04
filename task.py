# Bradley Fayyad-Reina BFR algorithm

import time
import numpy as np
import sys
import random
import itertools
from sklearn.cluster import KMeans


def create_cluster(K_Means):
    cluster = {}
    for i in K_Means.labels_:
        if i not in cluster:
            cluster[i] =1
        else:
            cluster[i] +=1
    return cluster

def get_total(S):
    s_all = 0
    for s in S:
        temp = len(S[s]["N"])
        s_all = s_all + temp
    return s_all

def get_cluster(S):
    s_cluster = len(S)
    return s_cluster

def renew_RS(RS_i,RS2):
   RS =[]
    
   for i in reversed(sorted(RS_i)): # reversed index
       RS.append(RS2[i])
        
   return RS

def get_distance(pnt,pnt2):
    centroid = pnt2["SUM"]/len(pnt2["N"])
    signature = pnt2["SUMSQ"]/len(pnt2["N"])-centroid**2
    z =(pnt - centroid)/signature
    mah = np.dot(z,z) ** (1/2)
    
    return mah

def get_result(data_dict,RS, DS, CS, result):
    for r in range(len(data_dict)):
        if r not in RS:
            for d in DS:
                if r in DS[d]["N"]:
                    result.append(str(r)+","+str(d)+"\n")
                    break
            for c in CS:
                if r in CS[c]["N"]:
                    result.append(str(r)+",-1\n")
                    break
        else:
            result.append(str(r)+",-1\n")
    return result


def write(output_file,result):
    with open(output_file,"w+") as f:
        for l in result:
            f.writelines(l)
      

if __name__ == '__main__':
    input_file = sys.argv[1]
    n_cluster = int(sys.argv[2])
    output_file = sys.argv[3]
    
    start = time.time()
  
    with open(input_file, "r") as f:
        lines = f.readlines()
    data = [line.strip().split(",") for line in lines]
    data = [(int(i[0]), tuple([float(j) for j in i[2:]])) for i in data]
    data_dict = dict(data)
    
    data_dict_vals = list(data_dict.values())
    data_dict_keys = list(data_dict.keys())
    data_dict_kv = zip(data_dict_vals,data_dict_keys)
    data_dict_2 = dict(data_dict_kv)
    data = list(map(lambda x: np.array(x), list(data_dict.values())))

    #print(data[1:3])
    
    # step 1: load 20% of the data randomly
    random.shuffle(data)
    length = len(data)*0.2
    length = int(length)
    data1 = data[0:length]
    #print(data1)
    # step 2: Run K-Means
    num = 5 # greater than 5 at least
    K_Means =KMeans(n_clusters= num * n_cluster).fit(data1)
    
    # step 3: Move all clusters that contain only one point to RS(outlier)
    #cluster = {}
    RS = []
    
    cluster = create_cluster(K_Means)
    
    RS_i = []
    
    for l in cluster:
        if cluster[l] < 20:
            for i, x in enumerate(K_Means.labels_):
                if x == l:
                    RS_i.append(i)
                
    for i in RS_i:
        RS.append(data1[i])
        
    RS_i2 = reversed(sorted(RS_i))
    
    for i in RS_i2:
        data1.pop(i)
    
    # step 4: K Means again
    K_Means = KMeans(n_clusters=n_cluster).fit(data1)
    # step 5: generate DS cluster
    DS ={}
    
    pair_zip  = zip(K_Means.labels_,data1)
    pair_cluster = list(pair_zip)
    for p in pair_cluster:
        item= p[0]
        item2 = p[1]
        item2_sq = item2**2
        if item in DS:
            pending = data_dict_2[tuple(item2)]
            DS[item]['N'].append(pending)
            DS[item]["SUM"] = DS[item]["SUM"] + item2
            DS[item]["SUMSQ"] = DS[item]["SUMSQ"] + item2_sq
        else:
            pending = data_dict_2[tuple(item2)]
            DS[item] = {}
            DS[item]["N"]=[pending]
            DS[item]["SUM"] =item2
            DS[item]["SUMSQ"] =item2_sq
     
    # step 6 Run K-Means on the points in the RS with a large K
    #print(RS)
    num_clusters2 = len(RS)-1
    
    K_Means = KMeans(n_clusters = num_clusters2).fit(RS)
    
    
    cluster = create_cluster(K_Means)
    temp_key = list(cluster.keys())
    temp_values = list(cluster.values())
    index2 = temp_values.index(2)
    
    RS_k = temp_key[index2]
    
    RS_i = []
    
    cluster_ky = list(cluster.keys())
    for i in cluster_ky:
        if i != RS_k:
            k_list = list(K_Means.labels_)
            RS_i.append(k_list.index(i))
            
    pair_zip  = zip(K_Means.labels_,RS)
    pair_cluster = tuple(pair_zip)
    
    CS = {}
    for p in pair_cluster:
        item = p[0]
        item2 = p[1]
        item2_sq = item2**2
        
        if (item == RS_k) & (item in CS):
            pending = data_dict_2[tuple(item2)]
            CS[item]['N'].append(pending)
            CS[item]["SUM"] = CS[item]["SUM"] + item2
            CS[item]["SUMSQ"] = CS[item]["SUMSQ"] + item2_sq
        if (item == RS_k) & (item not in CS):
            pending = data_dict_2[tuple(item2)]
            CS[item] = {}
            CS[item]["N"]=[pending]
            CS[item]["SUM"] =item2
            CS[item]["SUMSQ"] = item2_sq

    
    RS2 = RS.copy()
    
    RS = renew_RS(RS_i,RS2)   
    
    DS_all_str = str(get_total(DS))
    CS_all_str = str(get_total(CS))
    
    CS_cluster_str = str(get_cluster(CS))
    RS_all_str  = str(get_cluster(RS))
    
    result = ["The intermediate results:\n"]
    result.append("Round 1:" + DS_all_str+","+CS_cluster_str+","+CS_all_str+","+RS_all_str+"\n")
    
    # step 7 Load another 20% of the data randomly.
    times = [0,1,2,3]
    for t in times:
        if t!=3:
            data1 = data[length*(t+1):length*(t+2)]
        else:
            data1 = data[length*4:]
        
        # step 8: Mahalanobis distance DS; distance to nearst DS cluster: < 2 sqrt(d)
        DS_i = set()
        for i in range(len(data1)):
            mah_dict = dict()
            pnt = data1[i]
            for c in DS:
                mah_dict[c] = get_distance(pnt,DS[c])
            mah_dict_list = list(mah_dict.values())
            mah_final = min(mah_dict_list)
            
            for m in mah_dict:
                if mah_dict[m]==mah_final:
                    c = m
            sqrt_d = len(pnt)**(1/2)
            bar = 2 * sqrt_d
            if mah_final < bar:
                pntsq = pnt**2
                DS[c]["N"].append(data_dict_2[tuple(pnt)])
                DS[c]["SUM"] = DS[c]["SUM"] + pnt
                DS[c]["SUMSQ"] = DS[c]["SUMSQ"] + pntsq
                DS_i.add(i)
        # step 9 : same        
        CS_i = set()        
        for i in range(len(data1)):
            if i in DS_i:
                continue
            else:
                mah_dict = {}
                pnt = data1[i]
                for c in CS:
                    mah_dict[c] = get_distance(pnt,CS[c])
                mah_dict_list = list(mah_dict.values())
                mah_final = min(mah_dict_list)

                for m in mah_dict:
                    if mah_dict[m]==mah_final:
                        c = m
                sqrt_d = len(pnt)**(1/2)
                bar = 2 * sqrt_d
                if mah_final < bar:
                    pntsq = pnt**2
                    CS[c]["N"].append(data_dict_2[tuple(pnt)])
                    CS[c]["SUM"] = CS[c]["SUM"] + pnt
                    CS[c]["SUMSQ"] = CS[c]["SUMSQ"] + pntsq
                    CS_i.add(i)
                
        # step 10: assign non CS DS points to RS
        DS_CS_i = DS_i.union(CS_i)
        for i in range(len(data1)):
            if i not in DS_CS_i:
                RS.append(data1[i])
        
        num_clusters3 = len(RS)-1
        # step 11: K_Means on RS with K >=5
        K_Means = KMeans(n_clusters=num_clusters3).fit(RS)
     
        
        cluster = create_cluster(K_Means)
        
        temp_key = list(cluster.keys())
        temp_values = list(cluster.values())
        index2 = temp_values.index(2)
        
        RS_k = temp_key[index2]
 
        RS_i = []
       
        
        cluster_ky = list(cluster.keys())
        for i in cluster_ky:
            if i != RS_k:
                k_list = list(K_Means.labels_)
                RS_i.append(k_list.index(i))

        pair_zip  = zip(K_Means.labels_,RS)
        pair_cluster = tuple(pair_zip)
        CS = dict()
        
        for p in pair_cluster:
            item = p[0]
            item2 = p[1]
            item2_sq = item2**2
           
            if (item == RS_k) & (item in CS):
                pending = data_dict_2[tuple(item2)]
                CS[item]['N'].append(pending)
                CS[item]["SUM"] = CS[item]["SUM"] + item2
                CS[item]["SUMSQ"] = CS[item]["SUMSQ"] + item2_sq
            if (item == RS_k) & (item not in CS):
                pending = data_dict_2[tuple(item2)]
                CS[item] = {}
                CS[item]["N"]=[pending]
                CS[item]["SUM"] =item2
                CS[item]["SUMSQ"] = item2_sq

        RS2 = RS.copy()
        
        RS = renew_RS(RS_i,RS2)
        # step 12: merge CS cluster
        term = True
        while True:
            cluster_o = set(CS.keys())
            compare_list = list(itertools.combinations(list(CS.keys()), 2))
            for com in compare_list:
                val1 = com[0]
                val2 = com[1]
                ctr1 = CS[val1]["SUM"]/len(CS[val1]["N"])
                ctr2 = CS[val2]["SUM"]/len(CS[val2]["N"])
                sig1 = CS[val1]["SUMSQ"]/len(CS[val1]["N"])-(ctr1) ** 2
                sig2 = CS[val2]["SUMSQ"]/len(CS[val2]["N"])-(ctr2) ** 2
                z1 = (ctr1-ctr2)/sig1
                z2 = (ctr1-ctr2)/sig2
                m1 = np.dot(z1,z1) ** 0.5
                m2 = np.dot(z2,z2) ** 0.5
                min_mah = min(m1, m2)
                
                if min_mah < 2*(len(CS[val]["SUM"])** 0.5):
                    CS[val1]["N"] = CS[val1]["N"] + CS[val2]["N"]
                    CS[val1]["SUM"] = CS[val1]["SUM"] + CS[val2]["SUM"]
                    CS[val1]["SUMSQ"] = CS[val1]["SUMSQ"]+  CS[val2]["SUMSQ"]
                    CS.pop(val2)
                    term = False
                    break
                    
            cluster_n = set(CS.keys())
            if cluster_n == cluster_o:
                break
                
        CS_k = list(CS.keys())
        if t == 3:
            for ck in CS_k:
                mah_dict = {}
                for d in DS:
                    ctr1 = DS[d]["SUM"]/len(DS[d]["N"])
                    ctr2 = CS[ck]["SUM"]/len(CS[ck]["N"]) 
                    sig1 = DS[d]["SUMSQ"]/len(DS[d]["N"])-(ctr1) ** 2
                    sig2 = CS[ck]["SUMSQ"]/len(CS[ck]["N"])-(ctr2) ** 2
                    z1 = (ctr1-ctr2)/sig1
                    z2 = (ctr1-ctr2)/sig2
                    m1 = np.dot(z1,z1) ** 0.5
                    m2 = np.dot(z2,z2) ** 0.5
                    min_mah2 = min(m1, m2)
                    mah_dict[d] = min_mah2
                mah_dict_list = list(mah_dict.values())
                mah_final = min(mah_dict_list)
                for j in mah_dict:
                    if mah_dict[j] == mah_final:
                        c = j
                if mah_final < 2 * (len(CS[ck]["SUM"]) ** 0.5):
                    DS[c]["N"] = DS[c]["N"] + CS[ck]["N"]
                    DS[c]["SUM"] =DS[c]["SUM"]+ CS[ck]["SUM"]
                    DS[c]["SUMSQ"] =DS[c]["SUMSQ"]+ CS[ck]["SUMSQ"]
                    CS.pop(ck)
                    
        DS_all_str = str(get_total(DS))
        CS_all_str = str(get_total(CS))

        CS_cluster_str = str(get_cluster(CS))
        RS_all_str  = str(get_cluster(RS))

  
        result.append("Round "+ str(t+2)+":" + DS_all_str+","+CS_cluster_str+","+CS_all_str+","+RS_all_str+"\n")
    result.append("\nThe clustering results:\n")
    
    for d in DS:
        DS[d]["N"] = set(DS[d]["N"]) 
    for c in CS:
        CS[c]["N"] = set(CS[c]["N"]) 
    RS_final = set()
    for r in RS:
        RS_final.add(data_dict_2[tuple(r)])
        
  
    result = get_result(data_dict,RS_final,DS,CS,result)
    write(output_file,result)
    
    end =time.time()
    print("Duration:", end-start)

