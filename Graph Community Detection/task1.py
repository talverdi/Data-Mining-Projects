from pyspark import SQLContext, SparkContext
from pyspark import SparkConf
import time
import sys
import os
from pyspark.sql.types import *
from graphframes import *
from pyspark.sql import Row
import json
import itertools
from collections import defaultdict
from functools import reduce
from pyspark.sql.functions import col, lit, when
from graphframes import *
from pyspark.sql import *
from pyspark.sql.types import *
import sys
from pyspark.sql.functions import*


thresh=int(sys.argv[1])
input_path=sys.argv[2]
output_path=sys.argv[3]

start = time.time()

os.environ["PYSPARK_SUBMIT_ARGS"] = ("--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11 pyspark-shell")

sc_conf = SparkConf().setMaster('local[3]')
sc = SparkContext(conf = sc_conf)
sqlContext = SQLContext(sc)
sc.setLogLevel("error")
sc.setLogLevel("warn")

user_biz_file = sc.textFile(input_path)
header = user_biz_file.first()
user_biz_rdd = user_biz_file.filter(lambda row: row != header).map(lambda x: x.split(","))
user_biz_dict=user_biz_rdd.groupByKey().mapValues(set).collectAsMap()


#master_list=[]
edge_list=[]
node_list_temp=[]

distinct_user_list=user_biz_rdd.map(lambda x: x[0]).distinct().collect()
for i in range(len(distinct_user_list)):

    user_i=distinct_user_list[i]
    
    user_i_biz=user_biz_dict[user_i]
 
    other_users_list=distinct_user_list[0:i]+distinct_user_list[i+1:]
    for j in range(len(other_users_list)):
        
        user_j=other_users_list[j]
        
        user_j_biz=user_biz_dict[user_j]
        
        common_biz=set(user_i_biz).intersection(set(user_j_biz))
        if len(common_biz)>=thresh:
            #print(len(common_biz))
            edge_list.append((user_i,user_j))
            node_list_temp.append(user_i)
            node_list_temp.append(user_j)
    #master_list.append((edge_list,set(node_list_temp)))
    

node_list=list(set(node_list_temp))       
        
node_list_labeled=sc.parallelize(node_list).map(lambda x: Row(id = x)).collect()
#newl = [(i+1,node_list[i]) for i in range(len(node_list))]
vertex = sqlContext.createDataFrame(node_list_labeled)
edge = sqlContext.createDataFrame(edge_list, ["src", "dst"])
graph = GraphFrame(vertex, edge)

LPA = graph.labelPropagation(maxIter=5)
LPA_rdd = LPA.rdd.map(lambda x: (x['label'], x['id'])).groupByKey().map(lambda x: (x[0], sorted(list(x[1]))))
LPA_list = LPA_rdd.collect()


output=[]
write_to_file=[]
for i in LPA_list:
    output.append(i[1])
output=sorted(output,key=lambda x: x[0])   
output = sorted(output, key=lambda x: len(x))
for i in output:
    write_to_file.append("'"+"', '".join(i)+"'\n")
file = open(output_path, 'wt')
for line in write_to_file:
    file.write(line)

file.close()

end=time.time()
print("************************************")
print("************************************")
print("************************************")
print("************************************")

print("Duration: " + str(end-start))

print("************************************")
print("************************************")
print("************************************")
print("************************************")