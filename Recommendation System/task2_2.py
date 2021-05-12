import numpy as np
import json
import math
import pyspark
from pyspark import SparkContext
from pyspark import SparkConf
from operator import add
from pyspark import SparkContext, SparkConf, StorageLevel
import sys
import time
import csv
import xgboost as xgb


conf = SparkConf().setMaster("local[*]").set('spark.executor.memory','4g').set('spark.driver.memory','4g')
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")
start = time.time()

folder_path=sys.argv[1]
#train_in=folder_path+'/yelp_train.csv'
test_in=sys.argv[2]
output_path=sys.argv[3]

train_file = sc.textFile(folder_path+'/yelp_train.csv')
header = train_file.first()
train_rdd = train_file.filter(lambda x : x != header).map(lambda x: x.split(",")).map(lambda x: (x[0], x[1], float(x[2]))).persist()

test_file = sc.textFile(test_in)
header = test_file.first()
test_rdd = test_file.filter(lambda x : x != header).map(lambda x: x.split(",")).persist()

biz_data_rdd=sc.textFile(folder_path+'/business.json').map(json.loads).map(lambda x:(x['business_id'], (x['stars'], x['review_count'])))
user_data_rdd=sc.textFile(folder_path+'/user.json').map(json.loads).map(lambda x:(x['user_id'], (x['average_stars'], x['review_count'])))

biz_data_rdd_dict=biz_data_rdd.collectAsMap()
user_data_rdd_dict=user_data_rdd.collectAsMap()

users_in_train = train_rdd.map(lambda x: x[0]).distinct().collect()
biz_in_train = train_rdd.map(lambda x: x[1]).distinct().collect()
users_in_test = test_rdd.map(lambda x: x[0]).distinct().collect()
biz_in_test = test_rdd.map(lambda x: x[1]).distinct().collect()
users_union=list(set().union(users_in_train,users_in_test))
biz_union=list(set().union(biz_in_train,biz_in_test))

users_union_int = list(range(len(users_union)))
users_map = {l1:w1 for w1,l1 in zip(users_union_int,users_union)}

biz_union_int = list(range(len(biz_union)))
biz_map = {l1:w1 for w1,l1 in zip(biz_union_int,biz_union)}



train_pairs = np.array(train_rdd.map(lambda x: [x[0], x[1]]).collect())
train_output = np.array(train_rdd.map(lambda x: x[2]).collect())
augmented_input_train = []
for pair in train_pairs:
    user_id=pair[0]
    biz_id=pair[1]
    f1_train=users_map[user_id]
    f2_train=biz_map[biz_id]
    f3_train=biz_data_rdd_dict[biz_id][0]
    f4_train=biz_data_rdd_dict[biz_id][1]
    f5_train=user_data_rdd_dict[user_id][0]
    f6_train=user_data_rdd_dict[user_id][1]
    augmented_input_train.append([f1_train,f2_train,f3_train,f4_train,f5_train,f6_train])
train_input = np.array(augmented_input_train)



test_pairs = np.array(test_rdd.map(lambda x: [x[0], x[1]]).collect())
augmented_input_test = []
for entry in test_pairs:
    test_u=entry[0]
    test_b=entry[1]
    f1_test=users_map[test_u]
    f2_test=biz_map[test_b]
    f3_test=biz_data_rdd_dict[test_b][0]
    f4_test=biz_data_rdd_dict[test_b][1]
    f5_test=user_data_rdd_dict[test_u][0]
    f6_test=user_data_rdd_dict[test_u][1]
    augmented_input_test.append([f1_test,f2_test ,f3_test ,f4_test ,f5_test ,f6_test ])        
test_input = np.array(augmented_input_test)

model = xgb.XGBRegressor(learning_rate = 0.1, n_estimators= 700, max_depth=6, min_child_weight=6, nthread=6).fit(train_input, train_output)
output = model.predict(test_input)
end=time.time()
print("Duration: " + str(end-start))



result_pairs=test_pairs.tolist()
result_pairs_tuple=[tuple(i) for i in result_pairs]
result_prediction=output.tolist()
#result=[result_pairs_tuple[i]+(result_prediction[i],) for i in range(len(result_prediction))]

f = open(output_path, 'w')
header="user_id, business_id, prediction"
f.write(header)
f.write('\n')
for i in range(len(result_prediction)):    
    line=str(result_pairs_tuple[i][0])+","+str(result_pairs_tuple[i][1])+","+str(result_prediction[i])
    f.write(line)
    f.write('\n')
f.close()