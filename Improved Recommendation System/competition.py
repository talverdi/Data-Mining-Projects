########################################Comments#################################################
#Error Distribution
#>=0 and <1: 101891
#>=1 and <2: 33178
#>=2 and <3: 6182
#>=3 and <4: 793
#>=4: 0
#*************************************************************************************************
#RMSE : 0.9806157821659971
#Execution Time: 192.63155579566956


#Method Description: The recommendation system uses a single model (XGBRegressor) to predict the rating.  
# Improvement of Yelp Recommendation System:

# 1. Hyperparameter Tuning 
# I used GridSearchCV to tune the parameters of xgb model and select the best combination that yield the minimum RMSE.
# The parameters that I chose is the following: 
#learning_rate = 0.1, n_estimators= 600, max_depth=5, min_child_weight=4, nthread=4

#################################################################################################
#2. New features
# In HW3, only 6 features were used to train the model: 
# 1. user_id
# 2. business_id
# 3. business_avg
# 4. number_of_reviews_business
# 5. user_avg
# 6. number_of_reviews_user

# In the improved version, 6 more features were extracted from business.json, user.json to build the model. 

# 7. latitude
# 8. longitude
# 9. is_open
#10. useful
#11. cool
#12. fans

#################################################################################################




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

biz_data_rdd=sc.textFile(folder_path+'/business.json').map(json.loads).map(lambda x:(x['business_id'], (x['stars'], x['review_count'],x['latitude'],x['longitude'],x['is_open'])))
user_data_rdd=sc.textFile(folder_path+'/user.json').map(json.loads).map(lambda x:(x['user_id'], (x['average_stars'], x['review_count'],x['useful'],x['funny'],x['cool'],x['fans'])))

biz_data_rdd_dict=biz_data_rdd.collectAsMap()
user_data_rdd_dict=user_data_rdd.collectAsMap()


#biz_att=sc.textFile(folder_path+'/business.json').map(json.loads).map(lambda x:(x['business_id'],x['attributes']))
#biz_att_dict=biz_att.collectAsMap()

testY = np.array(test_rdd.map(lambda x: x[2]).collect())
y_true=[float(x) for x in testY]

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
                    
    f7_train=biz_data_rdd_dict[biz_id][2]
    f8_train=biz_data_rdd_dict[biz_id][3]
    f10_train=biz_data_rdd_dict[biz_id][4]
    
    f11_train=user_data_rdd_dict[user_id][2]
    f12_train=user_data_rdd_dict[user_id][3]
    f13_train=user_data_rdd_dict[user_id][4]
    f14_train=user_data_rdd_dict[user_id][5]
    
    augmented_input_train.append([f1_train,f2_train,f3_train,f4_train,f5_train,f6_train,f7_train,f8_train,f10_train,f11_train,f13_train,f14_train])
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
    f7_test=biz_data_rdd_dict[test_b][2]
    f8_test=biz_data_rdd_dict[test_b][3]
    f10_test=biz_data_rdd_dict[test_b][4]#is_open
    
    f11_test=user_data_rdd_dict[test_u][2]#useful
    f12_test=user_data_rdd_dict[test_u][3]#funny
    f13_test=user_data_rdd_dict[test_u][4]#cool
    f14_test=user_data_rdd_dict[test_u][5]#fans
                    
    augmented_input_test.append([f1_test,f2_test ,f3_test ,f4_test ,f5_test ,f6_test,f7_test,f8_test,f10_test,f11_test,f13_test,f14_test])        
test_input = np.array(augmented_input_test)

model = xgb.XGBRegressor(learning_rate = 0.1, n_estimators= 600, max_depth=5, min_child_weight=4, nthread=4).fit(train_input, train_output)
output = model.predict(test_input)

for i in range(len(output)):
    if output[i]> 5:
        output[i]=5
    elif  output[i]< 1 :
        output[i]=1



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




num=0
denom=0
abs_dif1=0
abs_dif2=0
abs_dif3=0
abs_dif4=0
abs_dif5=0
for i in range(len(output)):
    denom+=1
    abs_difference=abs(y_true[i]-output[i])
    diff=abs(y_true[i]-output[i])
    if abs_difference>=0 and abs_difference<1: 
        abs_dif1+=1
    if abs_difference>=1 and abs_difference<2: 
        abs_dif2+=1
    if abs_difference>=2 and abs_difference<3: 
        abs_dif3+=1
    if abs_difference>=3 and abs_difference<4: 
        abs_dif4+=1
    if abs_difference>=4: 
        abs_dif5+=1
    num+=(y_true[i]-output[i])**2

print("****************************************************")

print(">=0 and <1: ",abs_dif1)
print(">=1 and <2: ",abs_dif2)
print(">=2 and <3: ",abs_dif3)
print(">=3 and <4: ",abs_dif4)
print(">=4: ",abs_dif5)

print("****************************************************")
RMSE= (num/denom)**0.5
print("RMSE : ",RMSE)

end=time.time()
print("Duration: " + str(end-start))
