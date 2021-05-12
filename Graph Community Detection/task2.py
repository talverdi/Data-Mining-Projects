
import time
from pyspark import SparkConf, SparkContext
import json
import itertools
from collections import defaultdict
import sys



start = time.time()

SparkContext.setSystemProperty('spark.executor.memory', '7g')
SparkContext.setSystemProperty('spark.driver.memory', '7g')
sc = SparkContext(master="local[*]")
sc.setLogLevel("warn")
sc.setLogLevel("error")


thresh=int(sys.argv[1])
input_path=sys.argv[2]
btw_output_path=sys.argv[3]
com_output_path=sys.argv[4]

###################################################################################################################################################
user_biz_file = sc.textFile(input_path)
header = user_biz_file.first()
user_biz_rdd = user_biz_file.filter(lambda row: row != header).map(lambda x: x.split(","))
user_biz_dict=user_biz_rdd.groupByKey().mapValues(set).collectAsMap() 
###################################################################################################################################################


A_mat_dict={}
edge_list=[]
node_list_temp=[]
adj_nodes = defaultdict()

distinct_user_list=user_biz_rdd.map(lambda x: x[0]).distinct().collect()
for i in range(len(distinct_user_list)):

    user_i=distinct_user_list[i]
    
    user_i_biz=user_biz_dict[user_i]
    A_mat_dict[(user_i,user_i)]=0
     
    #other_users_list=distinct_user_list[0:i]+distinct_user_list[i+1:]
    other_users_list=distinct_user_list[i+1:]
    for j in range(len(other_users_list)):
        
        user_j=other_users_list[j]
        
        user_j_biz=user_biz_dict[user_j]
        
        common_biz=set(user_i_biz).intersection(set(user_j_biz))
        A_mat_dict[(user_i,user_j)]=0
        A_mat_dict[(user_j,user_i)]=0
        if len(common_biz)>=thresh:
            
            adj_nodes.setdefault(user_i, []).append(user_j)
            adj_nodes.setdefault(user_j, []).append(user_i)
            
            A_mat_dict[(user_i,user_j)]=1
            A_mat_dict[(user_j,user_i)]=1
            
            #print(len(common_biz))
            edge_list.append((user_i,user_j))
            node_list_temp.append(user_i)
            node_list_temp.append(user_j)
    #master_list.append((edge_list,set(node_list_temp)))
    
node_list=list(set(node_list_temp))         
###################################################################################################################################################    
def GNA_BTW(visited,parent_dict,depth_dict,w):
    btw_dict={}
    node_credit=dict(zip(reversed(visited), [1] * len(visited)))
    for node in reversed(visited[1:]):
        #print("node is :",node)
        num_paths_to_node=0
        for parent in parent_dict[node]:
            num_paths_to_node+=w[parent]*(depth_dict[node]==depth_dict[parent]+1)
        for parent in parent_dict[node]:
            if (depth_dict[node]==depth_dict[parent]+1):
                edge=tuple(sorted((node,parent)))
                #edge=tuple(sorted((str(node).encode("utf-8"),str(parent).encode("utf-8"))))
                #print("edge is:",edge)

                edge_fraction=float((node_credit[node]*w[parent])/num_paths_to_node)
                #node_fraction=float((node_credit[parent]*weights[parent])/num_paths_to_node)

                node_credit[parent]=node_credit[parent]+edge_fraction
                btw_dict[edge]=edge_fraction if edge not in btw_dict.keys() else btw_dict[edge]+edge_fraction
    return btw_dict
###################################################################################################################################################
def GNA_BFS(root_node,adj_nodes):

    visited = [] # List to keep track of visited nodes.
    queue = []     #Initialize a queue
    bfs_node_list=[]
    bfs_edge_list=[]
    
    parent_dict={}
    depth_dict={}
    w={}
    w[root_node]=1.0
    depth_dict[root_node]=0.0
    visited.append(root_node)
    queue.append(root_node)
    #print("queue is ", queue)
    while queue:
        current_node = queue.pop(0) #A,B
        bfs_node_list.append(current_node) #A,B
        for neighbour in adj_nodes[current_node]: #B,C
            if neighbour not in visited:
                visited.append(neighbour) #visited=[A,B,C,D,E]
                bfs_edge_list.append([current_node,neighbour]) #add (A,B), (A,C), (B,D), (B,E) 
                queue.append(neighbour) #queue=[B,C,D,E]
                parent_dict[neighbour]=[current_node]
                depth_dict[neighbour]=1+depth_dict[current_node]
                w[neighbour]=w[current_node]
            elif (neighbour in visited) & (neighbour!=root_node): # check if we need this for bds function
                    parent_dict[neighbour].append(current_node)
                    if(depth_dict[neighbour]==depth_dict[current_node]+1):
                        w[neighbour]+= w[current_node]
    return [visited,parent_dict,depth_dict,w]   
###################################################################################################################################################
def GNA_algorithm(root_node,adj_nodes):
    ####BFS
    bfs_output=GNA_BFS(root_node,adj_nodes)
    btw_output=GNA_BTW(bfs_output[0],bfs_output[1],bfs_output[2],bfs_output[3])
    btw_list=[]
    for edge in list(btw_output.keys()):
        #print (edge)
        btw_list.append([edge,float(btw_output[edge]/2)])
    return btw_list
node_list_rdd=sc.parallelize(node_list)
bt1_rdd= node_list_rdd.flatMap(lambda x: GNA_algorithm(x, adj_nodes))
bt2_rdd=bt1_rdd.reduceByKey(lambda x,y: (x+y)).sortByKey().map(lambda x: (x[1],x[0]))
betweenness_rdd=bt2_rdd.sortByKey(ascending=False).map(lambda x: (x[1],x[0]))



betweenness_final=betweenness_rdd.collect()
###################################################################################################################################################
def bfs_tree(adjc_vertices, root_node):
    visited_nodes = [] # List to keep track of visited nodes.
    queue = []     #Initialize a queue
    bfs_node_list=[]
    bfs_edge_list=[]
    visited_nodes.append(root_node)
    queue.append(root_node)
    #print("queue is ", queue)
    while queue:
        current_node = queue.pop(0) #A,B
#         print(current_node)
#         print("*******************************")
        bfs_node_list.append(current_node) #A,B
    #     print (res, end = " ") 
        for neighbour in adjc_vertices[current_node]: #B,C
            if neighbour not in visited_nodes:
                visited_nodes.append(neighbour) #visited=[A,B,C,D,E]
                bfs_edge_list.append([current_node,neighbour]) #add (A,B), (A,C), (B,D), (B,E) 
                queue.append(neighbour) #queue=[B,C,D,E]
    return (visited_nodes,bfs_edge_list)
###################################################################################################################################################

   


btw_dict=betweenness_rdd.collectAsMap()
max_value = max(btw_dict.values())
ed_to_remove=[k for k,v in btw_dict.items() if v == max_value]
#print("edge to remove:",ed_to_remove)
#print("****************************************************************")

for i in range(len(ed_to_remove)):
    adj_nodes[ed_to_remove[i][0]].remove(ed_to_remove[i][1])
    adj_nodes[ed_to_remove[i][1]].remove(ed_to_remove[i][0])
     
# edge_to_be_removed=list(btw_dict.keys())[0]
# adj_nodes[edge_to_be_removed[0]].remove(edge_to_be_removed[1])
# adj_nodes[edge_to_be_removed[1]].remove(edge_to_be_removed[0])

#initial_modul=-1
com_list_sorted=[]
modul_list=[]
com_master=[]
final_com=[]
modul=0.0
max_modul=-1000.0
#for iteration in range(34):
while modul>=max_modul:
#while modul>initial_modul:
    #initial_modul=modul
    ## detect communities 
    residual_G=node_list
    community_list=[]
    while len(residual_G)>0:
        node = residual_G[0]
        com = bfs_tree(adj_nodes,node) #returns nodes and edges between them 
        #re
        residual_G = [i for i in residual_G if i not in com[0]]
        community_list.append(com)
    com_master.append(community_list)

    #calculate modularity
    m=float(len(edge_list))
    q_list=[]
    com_num=1
    for com_num in range(len(community_list)):
        com_nodes=community_list[com_num][0]
    #     print(com_num)
    #     com_nodes=this_community[com_num][0]
        q=0
        for node_i in com_nodes:
            ki=float(len(adj_nodes[node_i]))
            for node_j in com_nodes:
                kj=float(len(adj_nodes[node_j]))
                
                A_ij=float(A_mat_dict[(node_i,node_j)]) 
                
                q+=A_ij-float((ki*kj)/(2*m))
        q_list.append(q)
    modul=float(sum(q_list)/(2*m))
    modul_list.append(modul)
    
    
    
    #print("iteration:",iteration)
    #print("************************modularity is:", modul)
    

    #recompute the betweennes and remove edges with highest btw
    bt1_rdd_1= node_list_rdd.flatMap(lambda x: GNA_algorithm(x, adj_nodes))
    bt2_rdd_1=bt1_rdd_1.reduceByKey(lambda x,y: (x+y)).sortByKey().map(lambda x: (x[1],x[0]))
    betweenness_rdd_1=bt2_rdd_1.sortByKey(ascending=False).map(lambda x: (x[1],x[0]))
    
    
    btw_dict=betweenness_rdd_1.collectAsMap()
    
    max_value = max(btw_dict.values())
    ed_to_remove=[k for k,v in btw_dict.items() if v == max_value]
    #print("edge to remove:",ed_to_remove)
    #print("****************************************************************")
    for i in range(len(ed_to_remove)):
        adj_nodes[ed_to_remove[i][0]].remove(ed_to_remove[i][1])
        adj_nodes[ed_to_remove[i][1]].remove(ed_to_remove[i][0])
    if modul>max_modul:
        max_modul=modul
        final_com=community_list
        #print("max modul :", max_modul)
        
#     edge_to_be_removed=list(btw_dict.keys())[0]
#     adj_nodes[edge_to_be_removed[0]].remove(edge_to_be_removed[1])
#     adj_nodes[edge_to_be_removed[1]].remove(edge_to_be_removed[0])



#community_modul_dict = dict(zip(modul_list,com_master))
#community_with_highest_modul = max(community_modul_dict.values(), key=lambda x : x[0])         
#print("**********************************")
#print("**********************************")
#print("max_modul: ",max_modul)
#print("len: " , len(final_com))

with open(btw_output_path, 'w') as fp:
    for item in betweenness_final:
        item=str(item).replace("u'", "'")
        #print(item) 
        fp.write(str(item)[1:][:-1] + '\n')
fp.close()



for i in final_com :
	item= sorted(i[0])
	com_list_sorted.append((item,len(item)))
com_list_sorted.sort()
com_list_sorted.sort(key=lambda x:x[1])


f = open(com_output_path, 'w')
num=0
for i in com_list_sorted:
	if(num==0):
		num=1
	else :
		f.write("\n")
	to_write= str(i[0]).replace("[","").replace("]","").replace("u'", "'")
	f.write(to_write)
f.close()


end=time.time()

print("**********************************")
print("**********************************")
print("Duration:", end-start)
print(len(final_com))
print("**********************************")
print("**********************************")