import json
import sys
import pyspark
from pyspark import SparkContext
from pyspark import SparkConf
import string

sc = SparkContext()
input_r= sc.textFile(sys.argv[1])
output_file = sys.argv[2]


data_r = input_r.map(lambda x: json.loads(x))

#Task1-A:The total number of reviews
a=data_r.count()

#Task1-B: The number of reviews in a given year, 2018
b=data_r.filter(lambda x: str(2018) in x['date']).count()

#Task1-C:The number of distinct users who wrote reviews
c=data_r.map(lambda x: (x['user_id'],[x['review_id']])).groupByKey().count()
#Task1-D Top 10 users who have the largest number of reviews and its count


d1=data_r.map(lambda x: (x['user_id'], x['review_id'])).groupByKey().map(lambda x: [str(x[0]), len(list(x[1]))])
d2=d1.sortBy(lambda x: (-x[1], x[0])).take(int(10))


#Task1-E:The number of distinct businesses that have been reviewed
e=data_r.map(lambda x: (x['business_id'],[x['review_id']])).groupByKey().count()

#Task1-F The top 10 businesses that had the largest numbers of reviews and the number of reviews they had 

f1=data_r.map(lambda x: (x['business_id'], x['review_id'])).groupByKey().map(lambda x: [str(x[0]), len(list(x[1]))])
f2=f1.sortBy(lambda x: (-x[1], x[0])).take(int(10))

output={}
output["n_review"]=a
output["n_review_2018"]=b
output["n_user"]=c
output["top10_user"]=d2
output["n_business"]=e
output["top10_business"]=f2

with open(output_file, 'w+') as f:
    json.dump(output,f)