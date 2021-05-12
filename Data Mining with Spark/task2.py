import json
import sys
import pyspark
from pyspark import SparkContext
from pyspark import SparkConf
import string
import time


sc = SparkContext()

input_r= sc.textFile(sys.argv[1])
output_file = sys.argv[2]
num_part = int(sys.argv[3])


#Idea of custom partition function taken from Oreily book : Page 70
#Example 4-27. Python custom partitioner
#import urlparse
#def hash_domain(url):
#return hash(urlparse.urlparse(url).netloc)



def custom_part(x):
    return hash(x)



start1 = time.time()
data_r = input_r.map(lambda x: json.loads(x))
data_r_map=data_r.map(lambda x: (x['business_id'], 1))
f = data_r_map.reduceByKey(lambda a, b: a+b).sortBy(lambda x: (-x[1], x[0])).take(10)
end1 = time.time()
res1=end1-start1
num_part1 = data_r_map.getNumPartitions()
num_items1 = data_r_map.glom().map(len).collect()



start2 = time.time()
data_r = input_r.map(lambda x: json.loads(x))
data_r_map=data_r.map(lambda x: (x['business_id'], 1))
data_r_map_2 = data_r_map.partitionBy(num_part,custom_part)
f = data_r_map_2.reduceByKey(lambda a, b: a+b).sortBy(lambda x: (-x[1], x[0])).take(10)
end2 = time.time()
res2=end2-start2
num_part2 = data_r_map_2.getNumPartitions()
num_items2 = data_r_map_2.glom().map(len).collect()




output1 = {}
output1["n_partition"] = num_part1
output1["n_items"] = num_items1
output1["exe_time"] = res1

output2 = {}
output2["n_partition"] = num_part2
output2["n_items"] = num_items2
output2["exe_time"] = res2

final={}
final["default"] = output1
final["customized"] = output2


with open(output_file, 'w+') as f:
    json.dump(final, f)
