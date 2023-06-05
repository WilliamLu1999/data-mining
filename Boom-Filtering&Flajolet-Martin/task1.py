import random
import csv
import sys
import time
import binascii
from blackbox import BlackBox


# the blackbox as a simulation of a data stream
# Bloom Filtering

def sampling(m,l):
 
    a = random.sample(range(1, m), l)
    b = random.sample(range(1, m), l)

    return a,b

def myhashs(s):
    # f(x)= (ax + b) % m or f(x) = ((ax + b) % p) % m
    result = []
    user_id = int(binascii.hexlify(s.encode('utf8')), 16)
    loop = 20
    m = 69997 # given
    p = 907 # any prime number
    a, b = sampling(m,loop)   
    for i in range(loop):
        result.append(((a[i] * user_id + b[i]) % p) % m)
    return result
    
def write(path,output):
    with open(path, 'w') as f:
        f.writelines("Time,FPR\n")
        f.writelines(output)
    
if __name__ == '__main__':
    
    input_filename = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_filename = sys.argv[4]
    user_final = set()
    temp = [0]*69997
    
    result = ""

    
    start = time.time()
    black_box = BlackBox()
   
    
    for i in range(num_of_asks):
        stream = black_box.ask(input_filename, stream_size)
        
        FP = 0
        TN = 0
        flag = 0
        for u in stream:
            hashed = myhashs(u)
            for h in hashed:
                
                if temp[h]!=1:
                    flag = 0
                    TN+=1
                    break
                if temp[h] ==1:
                    flag =1
            if u not in user_final and flag ==1:
                FP +=1
            user_final.add(u)
            
            
            for h in hashed:
                temp[h]=1
        if (FP+TN)==0:
            false_positive_rate = str(0.0) # make sure the denominator is not zero
        else:
            false_positive_rate = str(FP/(FP+TN))
            
        result +=  str(i) +','+false_positive_rate+ '\n'
        
    write(output_filename,result)
    end = time.time()
    print('Duration: ', end - start)