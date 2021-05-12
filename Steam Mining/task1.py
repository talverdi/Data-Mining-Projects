from blackbox import BlackBox
import binascii
import random
import sys
import time

start=time.time()
m=69997
input_file=sys.argv[1]
output_file=sys.argv[4]
num_of_asks=int(sys.argv[3])
stream_size=int(sys.argv[2])

hashList=((858,12598 ,13591),(4386,6672,13591),(1429,4367,13591))

num_hash=len(hashList)
bit_arr = [0] * m
final=[]
def myhashs(s):
    m=69997
    hashList=((858,12598 ,13591),(4386,6672,13591),(1429,4367,13591))
    x=int(binascii.hexlify(s.encode('utf8')),16)
    result=[]
    for hash_tuple in hashList:
        a=hash_tuple[0]
        b=hash_tuple[1]
        p=hash_tuple[2]
        f_s=((a * x + b) % p  % m)
        result.append(f_s)
    return result


seen_users=[]
bx=BlackBox()
fp_count=0.0

for ask in range(num_of_asks):
    stream_users=bx.ask(input_file, stream_size)
    for user in stream_users:
        user_hash=myhashs(user)
        #update bloom filter
        count=0
        for ind in user_hash:
            if bit_arr[ind]==1:
                count+=1
            else:
                bit_arr[ind]=1
                
        if (count==num_hash) and user not in set(seen_users):
            fp_count+=1
        seen_users.append(user)
    fp_ratio=float(fp_count/(stream_size*(ask+1)))
    
    final.append('{},{}\n'.format(ask,float(fp_ratio)))
#print(final)
with open(output_file, 'w') as fp:
    fp.write("Time,FPR\n")
    fp.writelines(final)
    

end=time.time()
print("**********************************************")
print("**********************************************")
print("Duration:",end-start)
print("**********************************************")
print("**********************************************")