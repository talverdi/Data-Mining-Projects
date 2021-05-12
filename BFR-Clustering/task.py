from sklearn.cluster import KMeans
import random
import sys
import numpy as np
from collections import defaultdict
from collections import Counter

input_file = sys.argv[1]
k_val=int(sys.argv[2])
output_file =sys.argv[3]

random.seed(30)

def MD_func(point,cluster_dict):
#     cluster_dict=DS_stats_dict
    dim=len(point)
    md_dict={}
    for key in cluster_dict:
        sigma_i=cluster_dict[key][4]
        C_i=cluster_dict[key][3]
        x_i=np.array(point)
        y_i_squared=((x_i-C_i)/sigma_i)**2
        d=sum(y_i_squared)**0.5
        md_dict[key]=d

    if min(md_dict.values()) < 2*(dim**0.5):
        cluster_assign=min(md_dict, key=md_dict.get)
    else:
        cluster_assign=-1
    
    return cluster_assign




my_data = open(input_file, 'r')
lines = [x.strip() for x in my_data.readlines()]
line_list = list()
for line in lines:
    b = list()
    b = line.split(',')
    split_list = list()
    for string in b:
        split_list.append(float(string))
    line_list.append(split_list)
my_data.close()        


data=line_list

data_size=len(data)
sample_size=int(data_size*0.2)
#STEP 1
#print("Step1")
sample=[]
sample_index=[]
for i in range(sample_size):
    random_id=random.sample(range(0, len(data)-1),1)   
    selected_row=data.pop(random_id[0])
    sample_index.append(selected_row[0])
    sample.append(selected_row[2:])
    
    
#STEP 2
#print("start")
kmeans = KMeans(n_clusters=(k_val * 10), random_state=0).fit(sample)

#STEP 3
#kmean_dict: (cluster_id,row id of data points in that cluster)
#print("Step3")
kmean_dict = defaultdict()
for i in range(len(sample)):
    key=kmeans.labels_[i]
    val=i
    kmean_dict.setdefault(key, []).append(val)
count_dict=Counter(kmeans.labels_)  
RS=[]
RS_index=[]
key_list=[]
RS_data_id_list=[]
rest_of_sample=[]
rest_of_sample_index=[]
for key in count_dict.keys():
    if count_dict[key]==1:
        data_id=kmean_dict[key][0]
        #selected_for_RS=sample.pop(data_id)
        RS.append(sample[data_id])
        RS_index.append(sample_index[data_id])
        RS_data_id_list.append(data_id)
    else:
        data_id=kmean_dict[key]
        for row_id in data_id:
            rest_of_sample.append(sample[row_id])
            rest_of_sample_index.append(sample_index[row_id])
        #selected_for_RS=sample.pop(data_id)
        
#STEP 4
#rest_of_sample_index=sample_index.remove(RS_data_index)
kmeans_step4 = KMeans(n_clusters=k_val, random_state=0).fit(rest_of_sample)


#STEP 5
#Generate DS Cluster
#print("Step5")
#DS_kmean_dict: {key:label, val:row_id}
DS_kmean_dict = defaultdict()
DS_index_dict=defaultdict()

for i in range(len(rest_of_sample)):
    key=kmeans_step4.labels_[i]
    row_id=i
    point_index=rest_of_sample_index[i]
    DS_kmean_dict.setdefault(key, []).append(row_id)
    DS_index_dict.setdefault(key, []).append(point_index)
    
#DS_count_dict=Counter(kmeans_step4.labels_) 
DS_stats_dict={}
for key in DS_kmean_dict.keys():
    # add a loop over all cluster
    point_ids=DS_kmean_dict[key]
    point_dim=len(rest_of_sample[0])
    sum_val=np.array([0]*point_dim)
    sumsq_val=np.array([0]*point_dim)
    for data_id in point_ids:
        sum_val=sum_val+np.array(rest_of_sample[data_id])
        squared_coord = [coord ** 2 for coord in rest_of_sample[data_id]]
        sumsq_val=sumsq_val+np.array(squared_coord)
    N_count=len(point_ids)
    term1=sumsq_val/N_count
    term2=[(x/N_count) ** 2 for x in sum_val] 
    std_val=np.sqrt(term1-term2)
    centroid=sum_val/N_count
    DS_stats_dict[key]=[N_count,sum_val,sumsq_val,centroid,std_val]
    # update the dictionary here and go to the next cluster
    
    
#STEP 6
k_step6 = min(k_val*5,int(0.85*len(RS)))
kmeans_step6 = KMeans(n_clusters=k_step6, random_state=0).fit(RS)


kmean_dict_step6 = defaultdict()
kmean_dict_step6_index = defaultdict()
for i in range(len(RS)):
    key=kmeans_step6.labels_[i]
    val=i
    kmean_dict_step6.setdefault(key, []).append(val)
    point_index=RS_index[i]    
    kmean_dict_step6_index.setdefault(key, []).append(point_index)

###########################################################################################  
count_dict_step6=Counter(kmeans_step6.labels_)  
RS_step6=[]
RS_index_step6=[]

CS_kmean_dict={}
CS_index_dict={}

key_list_step6=[]
RS_data_id_list_step6=[]
for key in count_dict_step6.keys():
    if count_dict_step6[key]==1:
        data_id=kmean_dict_step6[key][0]
        RS_step6.append(RS[data_id])
        RS_index_step6.append(RS_index[data_id])
        #RS_data_id_list.append(data_id)
    else:
        CS_kmean_dict[key]=kmean_dict_step6[key]
        CS_index_dict[key]=kmean_dict_step6_index[key]
        

################################################################               
CS_stats_dict={}
for key in CS_kmean_dict.keys():
    # add a loop over all cluster
    point_ids=CS_kmean_dict[key]
    point_dim=len(RS[0])
    sum_val=np.array([0]*point_dim)
    sumsq_val=np.array([0]*point_dim)
    for data_id in point_ids:
        sum_val=sum_val+np.array(RS[data_id])
        squared_coord = [coord ** 2 for coord in RS[data_id]]
        sumsq_val=sumsq_val+np.array(squared_coord)
    N_count=len(point_ids)
    term1=sumsq_val/N_count
    term2=[(x/N_count) ** 2 for x in sum_val] 
    std_val=np.sqrt(term1-term2)
    centroid=sum_val/N_count
    CS_stats_dict[key]=[N_count,sum_val,sumsq_val,centroid,std_val]        
    
    
f = open(output_file, "w+")
f.write("The intermediate results:\n")


DS_count=sum(DS_stats_dict[key][0] for key in DS_stats_dict.keys())
CS_cluster_count=len(CS_stats_dict)
CS_count=sum(CS_stats_dict[key][0] for key in CS_stats_dict.keys())
RS_count=len(RS_step6)
    
# Everything from here goes inside a loop
for iteration in range(1,6):
    #STEP 7
    sample=[]
    sample_index=[]
    if iteration==5:
        for i in range(len(data)):
# random_id=random.sample(range(0, len(data)-1),1)   
            selected_row=data[i]
            sample_index.append(selected_row[0])
            sample.append(selected_row[2:])
    else:
        f.write("Round " + str(iteration) +": " + str(DS_count) + "," + str(CS_cluster_count) + "," + str(CS_count) + "," + str(RS_count) + "\n")
        for i in range(sample_size):
            random_id=random.sample(range(0, len(data)-1),1)   
            selected_row=data.pop(random_id[0])
            sample_index.append(selected_row[0])
            sample.append(selected_row[2:])


    #STEP 8
    sample_not_in_DS=[]
    sample_not_in_DS_index=[]
    for i in range(len(sample)):
        point=sample[i]
        point_index=sample_index[i]
        cluster_assign=MD_func(point,DS_stats_dict)

        if cluster_assign==-1:
            sample_not_in_DS.append(point)
            sample_not_in_DS_index.append(point_index)
        else:

            DS_index_dict.setdefault(cluster_assign, []).append(point_index) #add point index to the corresponding cluster in DS
            DS_stats_dict[cluster_assign][0]+=1 #N_count
            DS_stats_dict[cluster_assign][1]+=point #sum_val
            squared_point = [coord ** 2 for coord in point] 
            DS_stats_dict[cluster_assign][2]+=squared_point #sumsq_val

            N_count=DS_stats_dict[cluster_assign][0]
            term1=DS_stats_dict[cluster_assign][2]/N_count
            term2=[(x/N_count) ** 2 for x in DS_stats_dict[cluster_assign][1]] 
            std_val=np.sqrt(term1-term2)
            centroid=DS_stats_dict[cluster_assign][1]/N_count

            DS_stats_dict[cluster_assign][3]=centroid
            DS_stats_dict[cluster_assign][4]=std_val        
            #update stats

    #STEP 9

    sample_not_in_DS_CS=[]
    sample_not_in_DS_CS_index=[]
    for i in range(len(sample_not_in_DS)):
        point=sample_not_in_DS[i]
        point_index=sample_not_in_DS_index[i]
        cluster_assign=MD_func(point,CS_stats_dict)

        if cluster_assign==-1:
            sample_not_in_DS_CS.append(point)
            sample_not_in_DS_CS_index.append(point_index)
        else:
            CS_index_dict.setdefault(cluster_assign, []).append(point_index) #add point index to the corresponding cluster in CS
            CS_stats_dict[cluster_assign][0]+=1 #N_count
            CS_stats_dict[cluster_assign][1]+=point #sum_val
            squared_point = [coord ** 2 for coord in point] 
            CS_stats_dict[cluster_assign][2]+=squared_point #sumsq_val

            N_count=CS_stats_dict[cluster_assign][0]
            term1=CS_stats_dict[cluster_assign][2]/N_count
            term2=[(x/N_count) ** 2 for x in CS_stats_dict[cluster_assign][1]] 
            std_val=np.sqrt(term1-term2)
            centroid=CS_stats_dict[cluster_assign][1]/N_count

            CS_stats_dict[cluster_assign][3]=centroid
            CS_stats_dict[cluster_assign][4]=std_val        
            #update stats

    #STEP 10
    RS_new=sample_not_in_DS_CS+RS
    RS_new_index=sample_not_in_DS_CS_index+RS_index

    #STEP 11
    k_step11 = min(k_val*5,int(0.85*len(RS_new)))
    kmeans_step11 = KMeans(n_clusters=k_step11, random_state=0).fit(RS_new)
    kmean_dict_step11 = defaultdict()
    kmean_dict_step11_index=defaultdict()

    for i in range(len(RS_new)):
        key=kmeans_step11.labels_[i]
        val=i
        point_index=RS_new_index[i]
        kmean_dict_step11.setdefault(key, []).append(val)
        kmean_dict_step11_index.setdefault(key, []).append(point_index)
    ###############################    
    new_CS_kmean_dict={}    
    new_CS_index_dict={}

    count_dict_step11=Counter(kmeans_step11.labels_)  
    RS_step11=[]
    RS_step11_index=[]
    key_list_step11=[]
    #RS_data_id_list_step11=[]
    for key in count_dict_step11.keys():
        if count_dict_step11[key]==1:
            data_id=kmean_dict_step11[key][0]
            RS_step11.append(RS_new[data_id])
            RS_step11_index.append(RS_new_index[data_id])
            #RS_data_id_list.append(data_id)
        else:
            new_CS_kmean_dict[key]=kmean_dict_step11[key]
            new_CS_index_dict[key]=kmean_dict_step11_index[key]
    ###############################################################        
    new_CS_stats_dict={}
    for key in new_CS_kmean_dict.keys():
        # add a loop over all cluster
        point_ids=new_CS_kmean_dict[key]
        point_dim=len(RS_new[0])
        sum_val=np.array([0]*point_dim)
        sumsq_val=np.array([0]*point_dim)
        for data_id in point_ids:
            sum_val=sum_val+np.array(RS_new[data_id])
            squared_coord = [coord ** 2 for coord in RS_new[data_id]]
            sumsq_val=sumsq_val+np.array(squared_coord)
        N_count=len(point_ids)
        term1=sumsq_val/N_count
        term2=[(x/N_count) ** 2 for x in sum_val] 
        std_val=np.sqrt(term1-term2)
        centroid=sum_val/N_count
        new_CS_stats_dict[key]=[N_count,sum_val,sumsq_val,centroid,std_val]   

    #STEP 12

    new_CS_unmerged={}
    new_CS_unmerged_index={}

    for key in new_CS_stats_dict:
        center=new_CS_stats_dict[key][3]
        cluster_assign=MD_func(center,CS_stats_dict)

        if cluster_assign==-1:
            new_CS_unmerged[key]=new_CS_stats_dict[key]
            new_CS_unmerged_index[key]=new_CS_index_dict[key]
        else:

            CS_index_dict.setdefault(cluster_assign, []).extend(new_CS_index_dict[key]) #add point index to cluster in CS
            CS_stats_dict[cluster_assign][0]+=new_CS_stats_dict[key][0] #N_count
            CS_stats_dict[cluster_assign][1]+=new_CS_stats_dict[key][1] #sum_val
            #squared_point = [coord ** 2 for coord in point] 
            CS_stats_dict[cluster_assign][2]+=new_CS_stats_dict[key][2] #sumsq_val
            N_count=CS_stats_dict[cluster_assign][0]
            term1=CS_stats_dict[cluster_assign][2]/N_count
            term2=[(x/N_count) ** 2 for x in CS_stats_dict[cluster_assign][1]] 
            std_val=np.sqrt(term1-term2)
            centroid=CS_stats_dict[cluster_assign][1]/N_count
            CS_stats_dict[cluster_assign][3]=centroid
            CS_stats_dict[cluster_assign][4]=std_val        
    #update stats
    CS_stats_dict.update(new_CS_unmerged)       
    CS_index_dict.update(new_CS_unmerged_index)
    
    DS_count=sum(DS_stats_dict[key][0] for key in DS_stats_dict.keys())
    CS_cluster_count=len(CS_stats_dict)
    CS_count=sum(CS_stats_dict[key][0] for key in CS_stats_dict.keys())
    RS_count=len(RS_step11)
    
    
 #STEP 13 Out of loop:

CS_unmerged={}
CS_unmerged_index={}

for key in CS_stats_dict:
    center=CS_stats_dict[key][3]
    cluster_assign=MD_func(center,DS_stats_dict)
    
    if cluster_assign==-1:
        CS_unmerged[key]=CS_stats_dict[key]
        CS_unmerged_index[key]=CS_index_dict[key]
    else:
        DS_index_dict.setdefault(cluster_assign, []).extend(CS_index_dict[key]) #add point index to cluster in DS
        DS_stats_dict[cluster_assign][0]+=CS_stats_dict[key][0] #N_count
        DS_stats_dict[cluster_assign][1]+=CS_stats_dict[key][1] #sum_val
        #squared_point = [coord ** 2 for coord in point] 
        DS_stats_dict[cluster_assign][2]+=CS_stats_dict[key][2] #sumsq_val
        
        N_count=DS_stats_dict[cluster_assign][0]
        term1=DS_stats_dict[cluster_assign][2]/N_count
        term2=[(x/N_count) ** 2 for x in DS_stats_dict[cluster_assign][1]] 
        std_val=np.sqrt(term1-term2)
        centroid=DS_stats_dict[cluster_assign][1]/N_count
        DS_stats_dict[cluster_assign][3]=centroid
        DS_stats_dict[cluster_assign][4]=std_val        
        #update stats

DS_count=sum(DS_stats_dict[key][0] for key in DS_stats_dict.keys())
CS_cluster_count=len(CS_stats_dict)
CS_count=sum(CS_stats_dict[key][0] for key in CS_stats_dict.keys())
RS_count=len(RS_step11)        
f.write("Round " + str(iteration) +": " + str(DS_count) + "," + str(CS_cluster_count) + "," + str(CS_count) + "," + str(RS_count) + "\n")
#print("Round " + str(iteration) +": " + str(DS_count) + "," + str(CS_cluster_count) + "," + str(CS_count) + "," + str(RS_count) + "\n")
f.write("The clustering results:\n")

DS_res=[]
for k in DS_index_dict:
	vs = DS_index_dict[k]
	for v in vs:
		DS_res.append((v, k))
RS_res=[]
for k in RS_step11_index:
# 	vs = CS_index_dict[k]
# 	for v in vs:
		RS_res.append((k, -1))
final_clusters = sorted(DS_res+RS_res, key=lambda x: float(x[0]))
for val, key in final_clusters:
	f.write(str(int(val)) + "," + str(key) + "\n")
    
f.close()