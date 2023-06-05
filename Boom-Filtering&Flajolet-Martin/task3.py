import random
import csv
import sys
import time
import binascii
from blackbox import BlackBox


def write(path,output):
    with open(path, 'w') as f:
        f.writelines("seqnum,0_id,20_id,40_id,60_id,80_id\n")
        f.writelines(output)


if __name__ == '__main__':
    input_filename = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_filename = sys.argv[4]
    
    start = time.time()
    random.seed(553)
    black_box = BlackBox()
    user_lst = []
    n = 0
    result = ""
    for i in range(num_of_asks):
        stream = black_box.ask(input_filename, stream_size)

        for u in stream:
            if i==0:
                n =100
                user_lst.append(u)
            if i != 0:
                p = random.random()
                n +=1
                temp = 100/n
                if temp>p:
                    num = random.randint(0,99)
                    user_lst[num] = u
                    
        result+= str(n)+','+str(user_lst[0])+','+str(user_lst[20])+','+str(user_lst[40])+','+str(user_lst[60])+','+ str(user_lst[80])+'\n'
        
    write(output_filename,result)
    
    end = time.time()
    print("Duration:", end-start)