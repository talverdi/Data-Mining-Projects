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

#Processing

train_file = sc.textFile(folder_path+'/yelp_train.csv')
header = train_file.first()
train_rdd = train_file.filter(lambda x : x != header).map(lambda x: x.split(",")).map(lambda x: (x[0], x[1], float(x[2]))).persist()

test_file = sc.textFile(test_in)
header = test_file.first()
test_rdd = test_file.filter(lambda x : x != header).map(lambda x: x.split(",")).persist()



#train_file = sc.textFile(train_in)
#header = train_file.first()
#train_rdd = train_file.filter(lambda x : x != header).map(lambda x: x.split(","))

users_in_train = train_rdd.map(lambda x: x[0]).distinct().collect()
biz_in_train = train_rdd.map(lambda x: x[1]).distinct().collect()


#test_file = sc.textFile(test_in)
#header = test_file.first()
#test_rdd = test_file.filter(lambda x : x != header).map(lambda x: x.split(","))


users_in_test = test_rdd.map(lambda x: x[0]).distinct().collect()
biz_in_test = test_rdd.map(lambda x: x[1]).distinct().collect()



# #check whether test users and train users have any difference
# A=set(users_in_train)
# B=set(users_in_test)
# A.difference(B)
# B.difference(A)
# users = list(set(users_in_train) | set(users_in_test))
# businesses = list(set(biz_in_train) | set(biz_in_test))


test_rdd_keys = test_rdd.map(lambda x: (x[0], x[1]))


all_biz_ratings = train_rdd.map(lambda x: (x[1], [(x[0], x[2])])).reduceByKey(lambda x, y: x + y)
my_dict = all_biz_ratings.map(lambda x: (x[0], dict(x[1])))
all_biz_ratings_dict=dict(my_dict.collect())

all_user_ratings = train_rdd.map(lambda x: (x[0], [x[1], x[2]])).groupByKey()
user_dict = all_user_ratings.map(lambda x: (x[0], dict(x[1])))
all_user_ratings_dict=dict(user_dict.collect())



biz_avg_dict={}
for key in all_biz_ratings_dict.keys():
    a_dict=all_biz_ratings_dict[key]
    values_a = [float(x) for x in a_dict.values()]
    avg_dict_a=sum(values_a)/len(values_a)
    biz_avg_dict[key]=avg_dict_a
    
user_avg_dict={}
for key in all_user_ratings_dict.keys():
    b_dict=all_user_ratings_dict[key]
    values_b = [float(x) for x in b_dict.values()]
    avg_dict_b=sum(values_b)/len(values_b)
    user_avg_dict[key]=avg_dict_b

k_neighbors=10000  

def item_based_CF(test_tuple, all_user_ratings_dict, all_biz_ratings_dict,biz_avg_dict,user_avg_dict):
     
    user,biz=test_tuple 
     # get all ratings by user u as a dict   
    if user in all_user_ratings_dict:
        ratings_for_this_user = {}
        ratings_for_this_biz={}
        ratings_for_this_user = all_user_ratings_dict[user]
        # get all rating for item i 
        if biz in all_biz_ratings_dict :
            ratings_for_this_biz = all_biz_ratings_dict[biz]     
            neighbors = set(ratings_for_this_user.keys()) 

            #####################################################################################  
            p_num=0
            p_denom=0
            # lets say we are predicting the rating for user u & item i (user U1, item B3)
            correlation=0
            w_list_pos=[]
            rating_list=[]
            correlation_list=[]
            for n in neighbors: # for each item in neighbor items:   B1, B2, B4
                n_all_ratings = {}
                # get all ratings of item n : # n_all_ratings 
                if n in all_biz_ratings_dict:    ###### we might be able to do it without this if
                    n_all_ratings = all_biz_ratings_dict[n] #for n=B1: {U1:2, U2:3, U4:5}  
                corated_ratings_list = []    
                corated_set=set(n_all_ratings).intersection(ratings_for_this_biz)
                for user_id in corated_set:
                    #tuple of ratings of user b for biz (n,y)
                    corated_ratings_list.append((float(n_all_ratings[user_id]), float(ratings_for_this_biz[user_id])))
                
                if len(corated_ratings_list)==0:
                    correlation=1
                else: 
                    score_1=[]
                    score_2=[]
                    diff_1=[]
                    diff_2=[]
                    avg_1=0
                    avg_2=0
                    for pair in corated_ratings_list:     
                        score_1.append(pair[0])
                        score_2.append(pair[1])
                    
                    #average_a = (sum(score_1) / len(score_1))
                    #average_b = (sum(score_2) / len(score_2))
                    num = 0
                    
                    d1 = 0
                    d2 = 0
                    for i in range(len(score_1)):
                        term_1 = (score_1[i] - (sum(score_1) / len(score_1)))
                        term_2 = (score_2[i] - (sum(score_2) / len(score_2)))
                        num += (term_1 * term_2)
                        d1 += (term_1 * term_1)
                        d2 += (term_2 * term_2)
                    denom = math.sqrt(d1) * math.sqrt(d2)
                    correlation= float(num/denom) if denom !=0 else 1.0
    #                 if len(score_1) > 0 :
    #                     avg_1=np.average(score_1) 
    #                 if len(score_2) > 0 :
    #                     avg_2=np.average(score_2)     

    #                 for score in score_1:
    #                     diff_1.append(score-avg_1)
    #                 for score in score_2:
    #                     diff_2.append(score-avg_2)

    #                 w_12_num=np.dot(diff_1,diff_2) 
    #                 w_12_denom=np.sqrt(np.dot(diff_1,diff_1))*np.sqrt(np.dot(diff_2,diff_2))
    #                 correlation=float(w_12_num/w_12_denom) if w_12_denom!=0 else 1
                       
    
                    #correlation = get_pearson_correlation(score_1, score_2)
                w_list_pos.append(max(correlation,0))
                n_rating = float(ratings_for_this_user[n])  # rating of item n=B1 by user u=U1 : 2
                rating_list.append(n_rating)
            # choose the k neighbor biz with max w
            if len(w_list_pos)> k_neighbors:
                arr = np.array(w_list_pos)
                max_ids=arr.argsort()[-k_neighbors:][::-1]
                w_list=[0 if i not in list(max_ids) else w_list_pos[i] for i in range(len(w_list_pos))]
            else:
                w_list=w_list_pos   
            p_num = np.dot(rating_list,w_list) #w_list_abs=[abs(w) for w in w_list]
            p_denom = sum(w_list)

            pred = p_num / p_denom 
            #we can revisit this part  
            if p_denom==0 :
                #print("inside first if")
                pred=user_avg_dict[user]
        else:
            pred=user_avg_dict[user]
    else:
        pred=biz_avg_dict[biz]

    if pred<1:
        new_pred = 1
    elif pred>5:
        new_pred = 5
    else:
        new_pred = pred

    #print("new_pred : ", new_pred)
    return user, biz, new_pred  

predictions = test_rdd_keys.map(lambda test_tuple: item_based_CF(test_tuple, all_user_ratings_dict, all_biz_ratings_dict,biz_avg_dict,user_avg_dict))
pred_list=predictions.collect()
y_pred=[pred_list[i][2] for i in range(len(pred_list))] 




biz_data_rdd=sc.textFile(folder_path+'/business.json').map(json.loads).map(lambda x:(x['business_id'], (x['stars'], x['review_count'])))
user_data_rdd=sc.textFile(folder_path+'/user.json').map(json.loads).map(lambda x:(x['user_id'], (x['average_stars'], x['review_count'])))

biz_data_rdd_dict=biz_data_rdd.collectAsMap()
user_data_rdd_dict=user_data_rdd.collectAsMap()

#users_in_train = train_rdd.map(lambda x: x[0]).distinct().collect()
#biz_in_train = train_rdd.map(lambda x: x[1]).distinct().collect()
#users_in_test = test_rdd.map(lambda x: x[0]).distinct().collect()
#biz_in_test = test_rdd.map(lambda x: x[1]).distinct().collect()

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


import numpy as np
alpha=0.2
beta=1-alpha
arr1=np.array(y_pred)
arr2=np.array(result_prediction)
arr3 = alpha*arr1 + beta*arr2



f = open(output_path, 'w')
header="user_id, business_id, prediction"
f.write(header)
f.write('\n')
for i in range(len(result_prediction)):    
    line=str(result_pairs_tuple[i][0])+","+str(result_pairs_tuple[i][1])+","+str(arr3[i])
    f.write(line)
    f.write('\n')
f.close()
