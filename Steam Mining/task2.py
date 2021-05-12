from blackbox import BlackBox
from collections import defaultdict
import binascii
import random
import numpy as np
import sys
import time

start=time.time()


input_file=sys.argv[1]
output_file=sys.argv[4]
num_of_asks=int(sys.argv[3])
stream_size=int(sys.argv[2])


m=69997

hashList=((858,12598 ,13591),(4386,6672,13591),(1429,4367,13591),
    (7, 100, 293),
     (11, 36, 353),
     (19, 225, 911),
     (3, 64, 409),
     (7, 225, 1321),
     (19, 16, 1109),
     (23, 64, 641),
     (23, 225, 479),
     (11, 16, 269),
     (3, 196, 1009),
     (31, 64, 739),
     (7, 100, 211)) 


num_hash=len(hashList)
len_hash_group=3
num_hash_groups=int(num_hash/len_hash_group)



def myhashs(s):
    m=69997
    hashList=((858,12598 ,13591),(4386,6672,13591),(1429,4367,13591),
    (7, 100, 293),
     (11, 36, 353),
     (19, 225, 911),
     (3, 64, 409),
     (7, 225, 1321),
     (19, 16, 1109),
     (23, 64, 641),
     (23, 225, 479),
     (11, 16, 269),
     (3, 196, 1009),
     (31, 64, 739),
     (7, 100, 211)) 
    x=int(binascii.hexlify(s.encode('utf8')),16)
    result=[]
    for hash_tuple in hashList:
        a=hash_tuple[0]
        b=hash_tuple[1]
        p=hash_tuple[2]
        f_s=((a * x + b) % p  % m)
        result.append(f_s)
    return result

def convert_to_binary(n):
    return int("{0:b}".format(int(n)))


predict_count=0.0
actual_count=0.0


final=[]
for ask in range(num_of_asks):
    bx=BlackBox()
    binary_output = defaultdict()
    stream_users=bx.ask(input_file, stream_size)
    distinct_count=len(set(stream_users))

    for user in stream_users:
        user_hash=myhashs(user)
        for h in range(len(user_hash)):
            bin_val=convert_to_binary(user_hash[h])
            binary_output.setdefault(h, []).append(bin_val)
            
    R_list=[]           
    for d in binary_output.values(): #iteration over all h 
        max_trailing_zeros=max([len(str(x))-len(str(x).rstrip('0')) for x in d])        
        R_list.append(max_trailing_zeros)
        
    group_avg_list=[]
    for group_id in range(num_hash_groups):
        this_group=R_list[len_hash_group*group_id:len_hash_group*(group_id+1)]
        this_group_estimate=[2**x for x in this_group ]
        group_avg_list.append(np.mean(this_group_estimate))
                        
    median_val=np.median(group_avg_list)
    
    final.append('{},{},{}\n'.format(ask,distinct_count,int(median_val)))

    

    
        
        
    predict_count+=median_val
    actual_count+=distinct_count
    

    #print(ask,distinct_count,int(median_val))
                 
with open(output_file, 'w') as fp:
        fp.write("Time,Ground Truth,Estimation\n")
        fp.writelines(final)
    
print("result: ", predict_count/actual_count)    

end=time.time()
print("**********************************************")
print("**********************************************")
print("Duration:",end-start)
print("**********************************************")
print("**********************************************")