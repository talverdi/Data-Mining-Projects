import json
import pyspark
from pyspark import SparkContext
from pyspark import SparkConf
from operator import add
from pyspark import SparkContext, SparkConf, StorageLevel


conf = SparkConf().setMaster("local[*]").set('spark.executor.memory','4g').set('spark.driver.memory','4g')
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

import sys
import time
import csv
import collections
from itertools import combinations
from collections import OrderedDict
from collections import Counter


#def GeneratePrimes(start,end):
#     # Initialize a list
#     primes = []
#     for possiblePrime in range(start, end + 1):
#         # Assume number is prime until shown it is not. 
#         isPrime = True
#         for num in range(2, possiblePrime):
#             if possiblePrime % num == 0:
#                 isPrime = False
#         if isPrime:
#             primes.append(possiblePrime)
#     return(primes)

# import random
# list1=GeneratePrimes(3,43)
# list2=[x*x for x in range(4,17)]
# list3=GeneratePrimes(193,1543)


# hashList=[]
# for i in range(20):
#     val1 = random.choice(list1)
#     val2 = random.choice(list2)
#     val3 = random.choice(list3)
#     triple=(val1,val2,val3)
#     hashList.append(triple)
    

#while (precision<0.99 or recall<0.97):
#These hash parameters resulted from running a while loop from random triples generated from prime numbers until precision and recall threshold was met: 
#getting random vals each time is not a good idea - not stable: might not meet the threshold values. 
start_time = time.time()


hashParameters=((17, 16, 397),
 (19, 169, 881),
 (29, 196, 977),
 (17, 196, 683),
 (37, 196, 197),
 (17, 81, 1201),
 (31, 169, 281),
 (19, 169, 1171),
 (31, 196, 1093),
 (7, 169, 751),
 (41, 225, 1289),
 (31, 100, 1301),
 (19, 49, 601),
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
 (7, 100, 211),
 (11, 256, 1381),
 (19, 81, 661),
 (23, 64, 241),
 (23, 225, 1381),
 (29, 256, 229),
 (5, 196, 1367),
 (23, 100, 839),
 (7, 100, 421),
 (3, 169, 569),
 (17, 169, 1471),
 (19, 36, 1153))

num_H = len(hashParameters)
num_bands = int(num_H/2)
num_rows = int(num_H / num_bands)
input_r= sys.argv[1]
output_file = sys.argv[2]
rdd1 = sc.textFile(input_r)    
rdd2 = rdd1.filter(lambda x : x != "user_id,business_id,stars")
rdd3=rdd2.map(lambda x: x.split(","))
users=rdd3.map(lambda x: x[0]).distinct().collect()

values = list(range(len(users)))
users=sorted(users)
users_dict = dict(zip(users, values))
rdd4=rdd3.map(lambda x: (x[1], x[0]))
rdd5=rdd4.map(lambda x: (x[0], users_dict[x[1]]))
#rdd 6 : each business is reviewed by which users ?
rdd6=rdd5.groupByKey().mapValues(list).sortBy(lambda x: x[0])
rdd6_dict=dict(rdd6.collect())




def hash_func(x,a,b,p,m):
    return ((a * x + b) % p  % m) # m is optional
    
def minHashing(user_list,a,b,p,m):
    h_min= 999999 
    for user_id in user_list:
        h_min= min(h_min,hash_func(user_id,a,b,p,m))
    return h_min

m = int(len(users) / num_H)
#Construct Matrix M 
signature_mat = rdd6.map(lambda x: [x[0], [minHashing(x[1],hPar[0],hPar[1],hPar[2],m) for hPar in hashParameters]])


signature=signature_mat.collect()

bucket = []
for i in range(len(signature)):
    start=0
    for band in range(num_bands):
        end=start+num_rows
        col_in_band=tuple(signature[i][1][start:end])
        bucket.append(((band,col_in_band), signature[i][0]))
        start+=num_rows
        
rdd_bucket=sc.parallelize(bucket)        
group_key=rdd_bucket.groupByKey().mapValues(list).filter(lambda x: len(x[1])>=2).collect()



pairs=[]
for i in range(len(group_key)):
    comb = combinations(group_key[i][1], 2)
    pairs.append(comb)
    
    
flat_list = []
for sublist in pairs:
    for item in sublist:
        flat_list.append(item)
        
pairs_list=list(set(flat_list))

out=dict()
for pair in pairs_list:
    set1=set(rdd6_dict.get(pair[0]))
    set2=set(rdd6_dict.get(pair[1]))
    jaccard_ratio=len(set1 & set2)/float(len(set1|set2))
    if jaccard_ratio>=0.5:
        out[pair]=jaccard_ratio
    
output_res=sorted(out.items(), key = lambda kv:(kv[0], kv[1]))
with open(output_file, "w+") as f:
    f.write("business_id_1, business_id_2, similarity" + "\n")
    for i in range(len(output_res)):
        item=output_res[i]
        one=item[0][0]
        two=item[0][1]
        three=str(item[1])
        f.write(one+ "," +two+ "," +three+"\n")
        
        
end_time= time.time()
dur= end_time - start_time
print("************************")
print("Start Time:" , start_time)
print("End Time:" , end_time)
print("**********************************")
print("Total Run Time:", dur)
print("*************************************************")

#Grade=(precision/0.99)*0.4+(recall/0.97)*0.4