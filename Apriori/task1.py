import findspark
findspark.init()
import pyspark
import json
import pyspark
from pyspark import SparkContext
from pyspark import SparkConf
from operator import add
configuration = SparkConf()
configuration.set("spark.driver.memory", "4g")
configuration.set("spark.executor.memory", "4g")
sc = SparkContext.getOrCreate(configuration)
import sys
import time
import csv
import collections
from itertools import combinations
from collections import OrderedDict
import collections
from collections import Counter


# MY FUNCTIONS
def filterTheDict(dictObj, callback):
    newDict = dict()
    # Iterate over all the items in dictionary
    for (key, value) in dictObj.items():
        # Check if item satisfies the given condition then add to new dict
        if callback((key, value)):
            newDict[key] = value
    return newDict

def candCountDict(baskets_list,citems):
    baskets_list=list(baskets_list)
    freqset_item_dic={}
    for i in range(len(citems)):
      if type(citems[i]) is int:
         
         key=tuple(sorted([citems[i]]))
      else:
        key=tuple(citems[i])

      count=0
      for j in range(len(baskets_list)):
        set1=set(key)
        set2=set(baskets_list[j])
        flag=set1.issubset(set2)
        # print("******")
        # print(j)
        # print(flag)
        if flag:
          count+=1
          
      freqset_item_dic[key]=count
    freqset_item_dic=freqset_item_dic.items()
    return freqset_item_dic

# this function returns the frequent single items for a given list of basket
def singleItems(basket,support):
    freqset_single_dic={}
    for b in basket:
        for elem in b:
            if elem in freqset_single_dic:
                freqset_single_dic[elem] = freqset_single_dic[elem] + 1
            else:
                freqset_single_dic[elem] = 1
    newDict = filterTheDict(freqset_single_dic, lambda elem: elem[1] >= support)
    mylist=list(newDict.keys())
    #freqset_single=[(x,) for x in mylist]
    return mylist

# this function returns the candidate pairs items given list of freq. single
def candidPairs(freqset_single):
  candidate_pairs=list(set(combinations(freqset_single,2)))
  return candidate_pairs

# this function returns the candidate items 
def candidItems(fitems,k):
    comb = list()
    fitems = list(fitems)
    for i in range(len(fitems)-1):
      for j in range(i+1, len(fitems)):
        a = fitems[i]
        b = fitems[j]
        if a[0:(k-2)] == b[0:(k-2)]:
          comb.append(list(set(a) | set(b)))
        else:
          break
    return comb

# this function returns the frequent item list (3, 4, ...)

# def freqItems(baskets_list,citems,threshold):
#     freqset_item_dic={}   
#     for i in range(len(citems)):
#         count=0
#         set1=set(citems[i])
#         candid = tuple(sorted(set1))
#         for j in range(len(baskets_list)):
#             set2=set(baskets_list[j])
#             flag=set1.issubset(set2)
#             if flag:
#                 count+=1   
#         if count>=threshold:
#             freqset_item_dic[candid]=count
#     freqset_item_dic_list=list(freqset_item_dic.keys())
#     return freqset_item_dic_list

def freqItems(baskets_list, citems, threshold):
    freqset_item_dic = {}
    for i in range(len(citems)):
        count = 0
        set1 = set(citems[i])
        candid = tuple(sorted(set1))

        for j in range(len(baskets_list)):
            set2 = set(baskets_list[j])
            flag = set1.issubset(set2)

            if flag:
                count += 1
            if count >= threshold:
                freqset_item_dic[candid] = count
                break

    freqset_item_dic_list = list(freqset_item_dic.keys())
    return freqset_item_dic_list



#def freqItems(baskets_list, citems, threshold):
#     freqset_item_dic = []
# 
#     for i in range(len(citems)):
#         count = 0
#         set1 = set(citems[i])
#         candid = tuple(sorted(set1))
# 
#         for j in range(len(baskets_list)):
# 
#             set2 = set(baskets_list[j])
# 
#             if set1.issubset(set2):
#                 count += 1
#             if count >= threshold:
#                 freqset_item_dic.append(candid)
#                 break
# 
#     # freqset_item_dic_list = list(freqset_item_dic.keys())
#     return freqset_item_dic



# apriori function
def apriori(baskets_list, support, num_baskets):
    baskets_list = list(baskets_list)
 
    p_ratio = (float(len(baskets_list))/float(num_baskets)) 
    threshold = support*p_ratio
 
    output = list()
 
    frequentSingletons = sorted(singleItems(baskets_list,threshold))
    freqset_single=[(x,) for x in frequentSingletons]
    output.extend(freqset_single)
 

    k=2
    frequentItems = set(frequentSingletons)
    while len(frequentItems) != 0:
        if k==2:

            candidateFrequentItems = candidPairs(frequentItems)
        else:
            candidateFrequentItems = candidItems(frequentItems, k)


        FrequentItems_2 = freqItems(baskets_list, candidateFrequentItems, threshold)
        output.extend(FrequentItems_2)
        frequentItems = list(set(FrequentItems_2))
        frequentItems.sort()
        k=k+1
    return output
# sc = SparkContext()
start = time.time()

#task1.py <case number> <support> <input_file_path> <output_file_path>


caseNum=int(sys.argv[1])
support=int(sys.argv[2])
input_path= sys.argv[3]
outputFile=sys.argv[4]


rdd = sc.textFile(input_path)
rdd = rdd.filter(lambda x : x != "user_id,business_id")
rdd = rdd.mapPartitions(lambda x : csv.reader(x))

if caseNum==1:
  allBaskets =rdd.map(lambda line: (line[0], [line[1]])).reduceByKey(lambda a,b: a+b)
  #outputFile = "Emelia_Talverdi_Task1.txt"
elif caseNum==2:
  allBaskets = rdd.map(lambda line: (line[1], [line[0]])).reduceByKey(lambda a,b: a+b)
  #outputFile = "Emelia_Talverdi_Task1.txt"
allBaskets = allBaskets.map(lambda x : x[1])

totalCount = allBaskets.count()
#SON Phase 1
#Map
#Reduce
map1=allBaskets.mapPartitions(lambda x: apriori(x,support,totalCount)).map(lambda x:(x,1))
red1=map1.reduceByKey(lambda x,y: (1)).keys().collect()
cands=sorted(red1, key = lambda item: (len(item), item))
map2 = allBaskets.mapPartitions(lambda x : candCountDict(x,red1))
red2 = map2.reduceByKey(lambda x,y: (x+y))
red2=red2.filter(lambda x: x[1]>=support)
red2=red2.keys().collect()
freqs=sorted(red2, key = lambda item: (len(item), item))

outFile = open(outputFile, "w")
outFile.write("Candidates:\n")
if len(cands) != 0:
    len1 = len(cands[0])
    word = str(cands[0]).replace(',', '')
    outFile.write(word)
    for i in range(1, len(cands)):
        len2 = len(cands[i])
        if len1 == len2:
            outFile.write(",")
        else:
            outFile.write("\n\n")

        if len2 == 1:
            word = str(cands[i]).replace(',', '')
        else:
            word = str(cands[i])
        outFile.write(word)
        len1 = len2
outFile.write("\n\n")
outFile.write("Frequent Itemsets:\n")
if len(freqs) != 0:
    len1 = len(freqs[0])
    word = str(freqs[0]).replace(',', '')
    outFile.write(word)
    for i in range(1, len(freqs)):
        len2 = len(freqs[i])
        if len1 == len2:
            outFile.write(",")
        else:
            outFile.write("\n\n")

        if len2 == 1:
            word = str(freqs[i]).replace(',', '')
        else:
            word = str(freqs[i])
        outFile.write(word)
        len1 = len2

print("**************************************")
print("**************************************")
print("**************************************")
end = time.time()
print("Duration:", end - start, " seconds")
print("**************************************")
print("**************************************")
print("**************************************")