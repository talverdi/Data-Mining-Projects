import json
import sys
import pyspark
from pyspark import SparkContext
from pyspark import SparkConf
import string
import itertools
import operator
from itertools import islice
import time

sc = SparkContext()
output_file_A = sys.argv[3]
output_file_B = sys.argv[4]



start1 = time.time()
path1=sys.argv[1]
path2=sys.argv[2]


rev = []
for i in open(path1):
    rev.append(json.loads(i))
bus = []
for i in open(path2):
    bus.append(json.loads(i))

from collections import defaultdict

rev_dict = defaultdict()
for i in rev:
    key = i['business_id']
    value = i['stars']
    rev_dict.setdefault(key, []).append(value)
from collections import defaultdict

bus_dict = defaultdict()
for i in bus:
    key = i['business_id']
    value = i['city']
    bus_dict.setdefault(key, []).append(value)



fianl_dict = defaultdict()
for k1 in rev_dict:
    if k1 in bus_dict.keys():
        for val1 in bus_dict[k1]:
            for val2 in rev_dict[k1]:
                fianl_dict.setdefault(val1, []).append(val2)

keys = fianl_dict.keys()
vals = list(map(lambda x: sum(fianl_dict[x]) / len(fianl_dict[x]), fianl_dict))
final = dict(zip(keys, vals)) #make a dictionary form keys and vals


#ddd = sorted(final.items(),key=lambda t: (-t[1], t[0]), reverse=0)


num=10

my_list=[(k,v) for k,v in final.items()]

for mx in range(len(my_list)-1, -1, -1):
    swapped = False
    for i in range(mx):
        if my_list[i][1] < my_list[i+1][1]:
            my_list[i], my_list[i+1] = my_list[i+1], my_list[i]
            swapped = True
    if not swapped:
        break
        
B_dict=sorted(my_list, key=lambda x:(-x[1],x[0]))

itemDict = {item[0]: item[1] for item in B_dict}


#B_dict=dict(ddd[:num])
#B_dict = sorted(B_dict.items(),key=lambda t: (-t[1], t[0]), reverse=0)

end1 = time.time()
result1=end1 - start1

start2 = time.time()


input_r= sc.textFile(sys.argv[1])
input_b=sc.textFile(sys.argv[2])
data_r = input_r.map(lambda x: json.loads(x))
data_b = input_b.map(lambda x: json.loads(x))
bb = data_b.map(lambda x: (x['business_id'],x['city']))
rr = data_r.map(lambda x: (x['business_id'],x['stars']))

rdd_join = bb.join(rr)
rdd_join_2=rdd_join.map(lambda x:x[1])
val = rdd_join_2.map(lambda x: (x[0].split(','),x[1]))
val2=val.flatMap(lambda x: [(value,x[1]) for value in x[0]])
val3=val2.map(lambda x: (x[0].strip(),x[1]))
aTuple = (0,0)
val_avg=val3.aggregateByKey(aTuple, lambda a,b: (a[0] + b, a[1] + 1),lambda a,b: (a[0] + b[0], a[1] + b[1]))
finalResult = val_avg.mapValues(lambda v: v[0]/v[1])
A=finalResult.sortByKey().takeOrdered(10, key = lambda x: -x[1])
end2 = time.time()
result2=float(end2 - start2)

list_of_tuples = A
f = open(output_file_A, 'w')
f.write("city,stars\n")
for t in list_of_tuples:
    line = ','.join(str(x) for x in t)
    f.write(line + '\n')
f.close


final_time={}
final_time["m1"] = result1
final_time["m2"] = result2


with open(output_file_B, 'w+') as f:
    json.dump(final_time, f)
