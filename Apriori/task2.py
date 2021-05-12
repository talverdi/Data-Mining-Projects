import findspark
findspark.init()
import pyspark
import json
import string
import pyspark
from pyspark import SparkContext
from pyspark import SparkConf
from operator import add
from pyspark import SparkContext, SparkConf, StorageLevel
import sys
import time
import csv
import collections
from itertools import combinations
from collections import OrderedDict
import collections
from collections import Counter


    
configuration = SparkConf()
configuration.set("spark.driver.memory", "4g")
configuration.set("spark.executor.memory", "4g")
sc = SparkContext.getOrCreate(configuration)


def filterTheDict(dictObj, callback):
    newDict = dict()
    # Iterate over all the items in dictionary
    for (key, value) in dictObj.items():
        # Check if item satisfies the given condition then add to new dict
        if callback((key, value)):
            newDict[key] = value
    return newDict


# MY FUNCTIONS
import collections
from collections import Counter


# this function retruns a dictionary of candid items and their count
def candCountDict(baskets_list, citems):
    baskets_list = list(baskets_list)
    freqset_item_dic = {}
    for i in range(len(citems)):
        if type(citems[i]) is int:

            key = tuple(sorted([citems[i]]))
        else:
            key = tuple(citems[i])

        count = 0
        for j in range(len(baskets_list)):
            set1 = set(key)
            set2 = set(baskets_list[j])
            flag = set1.issubset(set2)
            # print("******")
            # print(j)
            # print(flag)
            if flag:
                count += 1

        freqset_item_dic[key] = count
    freqset_item_dic = freqset_item_dic.items()
    return freqset_item_dic




# this function returns the frequent single items for a given list of basket
def singleItems(basket, support):
    freqset_single_dic = {}
    for b in basket:
        for elem in b:
            if elem in freqset_single_dic:
                freqset_single_dic[elem] = freqset_single_dic[elem] + 1
            else:
                freqset_single_dic[elem] = 1
    newDict = filterTheDict(freqset_single_dic, lambda elem: elem[1] >= support)
    mylist = list(newDict.keys())
    # freqset_single=[(x,) for x in mylist]
    return mylist


def genCandidate(input_list, k):
    L = list(map(lambda x: set(x), input_list))
    #print(L)
    #print(len(L))
    candidate = []
    
    comb=combinations(range(len(L)), 2)
    for pair in comb:
        #print(pair)
        i=pair[0]
        j=pair[1]
        # for each tuple in comb
        # i is the first element 
        # j is the second element
        if len(L[i] | L[j]) == k:
            candidate.append(tuple(sorted(L[i] | L[j])))
    return candidate



# this function returns the frequent item list (3, 4, ...)
def freqItems(basket, ck, support):
    C1 = [tuple(x) for x in ck]
    cnt = {}
    for i in basket:
        for c in C1:
            if (set(c).issubset(i)):
                if c in cnt:
                    cnt[c] += 1
                else:
                    cnt[c] = 1
    freq_item = []
    for key in cnt:
        if cnt[key] >= support:
            freq_item.append(key)
    return freq_item


def apriori2(baskets_list, support, num_baskets):
    baskets_list = list(baskets_list)

    p_ratio = (float(len(baskets_list)) / float(num_baskets))
    threshold = support * p_ratio

    output = list()

    frequentSingletons = sorted(singleItems(baskets_list, threshold))
    freqset_single = [(x,) for x in frequentSingletons]
    output.extend(set(freqset_single))

    k = 2
    frequentItems = set(frequentSingletons)

    frequentItems = [(x,) for x in frequentItems]

    
    while len(frequentItems) != 0:
        #print("start ---> k :", k)
        #print(frequentItems)
        #if k == 2:

        candidateFrequentItems = set(genCandidate(frequentItems, k))
        # candidateFrequentItems = candidPairs(frequentItems)
        #else:
            #candidateFrequentItems = set(genCandidate(frequentItems, k))
        # candidateFrequentItems = candidItems(frequentItems, k)
        #print(" cand : ", candidateFrequentItems)
        FrequentItems_2 = set(freqItems(baskets_list, candidateFrequentItems, threshold))
        output.extend(FrequentItems_2)
        frequentItems = list(set(FrequentItems_2))
        frequentItems.sort()
        #print(" end : ", frequentItems)
        k = k + 1
    return output



start = time.time()



kval = int(sys.argv[1])
# caseNum = int(sys.argv[1])
support = int(sys.argv[2])
input_path = sys.argv[3]
outputFile = sys.argv[4]

partition_number=6

rdd1 = sc.textFile(input_path,partition_number)
header = rdd1.first() 
rdd2 = rdd1.filter(lambda x: x != header)
rdd3 = rdd2.mapPartitions(lambda x: csv.reader(x))

rdd4 = rdd3.map(lambda line: (line[0], line[1],line[5]))
rdd5=rdd4.map(lambda line: (line[0]+"-"+line[1],line[2].lstrip('0')))  


        
allBaskets = rdd5.groupByKey().mapValues(lambda x: list(set(x)))
allBaskets = allBaskets.map(lambda x: x[1])
allBaskets = allBaskets.filter(lambda x: len(x) > kval)
totalCount = allBaskets.count()

# SON Phase 1
# Map
# Reduce
map1 = allBaskets.persist(StorageLevel.DISK_ONLY)
#print("*****************after persist*********************")
map1 = map1.mapPartitions(lambda x: apriori2(x, support, totalCount)).map(lambda x: (x, 1))


red1 = map1.reduceByKey(lambda x, y: (1)).keys().collect()
#print("*****************after collect*************************")

cands = sorted(red1, key=lambda item: (len(item), item))
#cands=[cands[i][0].lstrip('0') for i in range(len(cands))]
#cands=[tuple(map(str, sub.split(', '))) for sub in cands]


map2 = allBaskets.mapPartitions(lambda x: candCountDict(x, red1))
red2 = map2.reduceByKey(lambda x, y: (x + y))
red2 = red2.filter(lambda x: x[1] >= support)
red2 = red2.keys().collect()
freqs = sorted(red2, key=lambda item: (len(item), item))
#freqs=[freqs[i][0].lstrip('0') for i in range(len(freqs))]
#freqs=[tuple(map(str, sub.split(', '))) for sub in freqs]


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

end = time.time()
print("**************************************")
print("**************************************")
print("**************************************")
print("Duration:", end - start, " seconds")
print("**************************************")
print("**************************************")
print("**************************************")
