import random
import csv
import sys
import time
import binascii
from blackbox import BlackBox


# Flajolet-Martin algorithm
# 300
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
    a,b = sampling(m,loop)
    for i in range(loop):
        result.append(((a[i] * user_id + b[i]) % p) % m)
    return result

        
def write(path,output):
    with open(path, 'w') as f:
        f.writelines("Time,Ground Truth,Estimation\n")
        f.writelines(output)
        
if __name__ == '__main__':
    input_filename = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_filename = sys.argv[4]
    
    start = time.time()
    black_box = BlackBox()
    ground_sum = 0
    estimate_sum = 0
    loop = 20
    result = ""
    
    for i in range(num_of_asks):
        temp = [0] * loop
        stream = black_box.ask(input_filename, stream_size)
        stream_len = len(set(stream))
        stream = set(stream)
        estimate = 0
        ground_sum += stream_len
        for u in stream:
            hashed = myhashs(u)
            for k in range(loop):
                binary = bin(hashed[k]).split("1")[-1]
                val = max(len(binary),temp[k])
                temp[k] = val
                
        for l in temp:
            estimate += 2**l
            
        
        estimate_sum += round(estimate/loop)
        
        
        result += str(i) +','+str(ground_sum)+','+str(estimate_sum) + '\n'
        
    write(output_filename,result)
    end = time.time()
    print('Duration: ', end - start)