import numpy as np
from torch.utils.data import Dataset
import torch
import os
from collections import defaultdict as ddict
import sys
import re

relative_path = ""

def get_id(datapath):
    toid = {}
    # idto = {}
    dict = open(os.path.join(relative_path, datapath),'r')
    lines = dict.readlines()
    for line in lines:
        id, en = line.strip().split()
        toid[en] = id
        # idto[id] = en
    dict.close()
    return toid


def get_num(en2id, re2id):
    triple = []
    enti_num = {}
    # rela_num = {}
    with open(os.path.join(relative_path,"train.txt")) as fin:
        for line in fin:
            # total_num += 1
            h, r, t = line.strip().split()
            triple.append((en2id[h],re2id[r],en2id[t]))
            enti_num[en2id[h]] = enti_num.get(en2id[h], 0) + 1
            # rela_num[re2id[r]] = rela_num.get(re2id[r], 0) + 1
            enti_num[en2id[t]] = enti_num.get(en2id[t], 0) + 1
        fin.close()

    with open(os.path.join(relative_path,"valid.txt")) as fin:
        for line in fin:
            # total_num += 1
            h, r, t = line.strip().split()
            triple.append((en2id[h],re2id[r],en2id[t]))
            enti_num[en2id[h]] = enti_num.get(en2id[h], 0) + 1
            # rela_num[re2id[r]] = rela_num.get(re2id[r], 0) + 1
            enti_num[en2id[t]] = enti_num.get(en2id[t], 0) + 1
        fin.close()
    
    return enti_num, triple
       
def save_triple(triple, file_path = "amie_datapy.tsv"):
    path_tmp = os.path.join(relative_path, "relation_classify") #dataset/FB15K237/relation_classify
    if not os.path.exists(path_tmp):
        os.makedirs(path_tmp)
    with open( os.path.join(path_tmp, file_path),"w") as file:
        for (h,r,t) in triple:
            file.write(h)
            file.write("\t")
            file.write(r)
            file.write("\t")
            file.write(t)
            file.write("\t")
            file.write(".")
            file.write("\n")
        file.close()
    print(os.path.join(path_tmp, file_path))
    return os.path.join(path_tmp, file_path) #dataset/FB15K237/relation_classify/amie_datapy.tsv

def AMIE(source_path, minhc=0.01, minc=None, minpca=None, maxad=3):
    '''AMIE: using AMIE for rule mining in java (https://github.com/lajus/amie)
        Attributes:
            source_path    : the input file.
            target_dir     : the output path.
            minhc          : min hc.
            minc           : min std confidence.
            minpca         : min pca confidence.
    '''

    amie_arg = "java -XX:-UseGCOverheadLimit -Xmx4G -jar " + os.path.join(relative_path,"../amie3.jar") + " " + source_path

    target_dir = ""

    if minhc:
        if minhc != 0.01:
            amie_arg += " -minhc "
            amie_arg += str(minhc)
        target_dir = "minhc_"+str(minhc)
    if minc:
        amie_arg += " -minc "
        amie_arg += str(minc)
        target_dir = target_dir+"_minc_"+str(minc)
    if minpca:
        amie_arg += " -minpca "
        amie_arg += str(minpca)
        target_dir = target_dir+"_minpca_"+str(minpca)
    if maxad:
        amie_arg += " -maxad "
        amie_arg += str(maxad)
        target_dir = target_dir+"_maxad_"+str(maxad)
    
    print(target_dir)

    target_dir = os.path.join(relative_path,"relation_classify",target_dir) #dataset/FB15K237/relation_classify/minhc_0.5_minpca_0.8_maxad_4
    if os.path.exists(target_dir):
        print("the same order has been executed before")
        return target_dir
    else:
        os.makedirs(target_dir)

    if os.path.exists(os.path.join(target_dir, "result.txt")):
        os.remove(os.path.join(target_dir, "result.txt"))
    amie_arg += " > " + os.path.join(target_dir, "result.txt")
    print(amie_arg)
    os.system(amie_arg)
    return target_dir #dataset/FB15K237/relation_classify/minhc_0.5_minpca_0.8_maxad_4

def classify_relation(target_dir):

    set_rela_symmetric   = set()
    set_rela_inverse     = set()
    set_rela_multiple    = set()
    set_rela_compose2    = set()
    set_rela_compose3    = set()

    miningfile = open(os.path.join(target_dir, "result.txt"), "r")
    lines = miningfile.readlines()
    for line in lines:
        if(line[0]!='?'):
            continue
        else:
            ls = re.findall(r"(\d+\.?\d*|\?[a-z])",line)
            if len(ls) == 13:  # r1->r2
                h_1, r_1, t_1, h_2, r_2, t_2 = ls[:6]
                #   symmetric
                if r_1==r_2 and h_1==t_2 and h_2==t_1:
                    set_rela_symmetric.add(r_2)
                #   set_rela_inverse
                if r_1!=r_2 and h_1==t_2 and h_2==t_1:
                    set_rela_inverse.add(r_2)
                #   multiple
                if r_1!=r_2 and h_1==h_2 and t_1==t_2:
                    set_rela_multiple.add(r_2)
            elif len(ls) == 16: # r1 + r2 -> r3
                #   compose
                if ls[1]==ls[7] and ls[4]==ls[7] :
                    continue
                set_rela_compose2.add(ls[7])
            elif len(ls) == 19: # r1 + r2 + r3 -> r4
                # a r_1 b c r_2 d e r_3 f g r_4 h
                # 0   1 2 3   4 5 6   7 8 9  10 11
                if ls[1]==ls[7] and ls[4]==ls[7] and ls[7]==ls[10] :
                    continue
                set_rela_compose3.add(ls[10])
    miningfile.close()
    return set_rela_symmetric, set_rela_inverse, set_rela_multiple, set_rela_compose2, set_rela_compose3

def classify_triples(re2id, set_rela):
    pattern_list = []
    # set_list = set()
    testfile = open(os.path.join( relative_path, "test_copy.txt"), "r")
    lines = testfile.readlines()
    for content in lines:
        h,r,t = content.strip().split()
        if re2id[r] in set_rela:
            pattern_list.append((h,r,t))
            # set_list.add((h,r,t))
    testfile.close()
    return pattern_list

def entity_frequency(target_dir, test_file, list_test, en2id, enti_num, num_constrain = [0,5,10,15,20]):
	print(test_file)
	if os.path.exists(os.path.join(target_dir, test_file)):
		print("the same order has been executed before")
	else:
		os.makedirs(os.path.join(target_dir, test_file))
	
	for num in num_constrain:
		test = open( os.path.join(target_dir, test_file, "num_constrain_"+str(num)+".txt"), "w")
		cnt = 0
		for tuple in list_test:
			h, r, t = tuple
			'''constrain for num'''
			if num != 0 and ( en2id[h] not in enti_num.keys() or enti_num[en2id[h]] < num or \
			   en2id[t] not in enti_num.keys() or enti_num[en2id[t]] < num ) :
				continue 
			cnt = cnt + 1
			test.write(h)
			test.write('\t')
			test.write(r)
			test.write('\t')
			test.write(t)
			test.write('\n')
		test.close()
		print(str(num)+"\t"+str(cnt))

def similarity(list_all):
    matrix = np.zeros((len(list_all),len(list_all)))
    for i, r1 in enumerate(list_all):
        for j, r2 in enumerate(list_all):
            matrix[i][j] = len(r1&r2)/len(r1)
    print(matrix)
    return matrix

def main(PCA=0.8, HC=0.5):
    en2id = get_id(datapath = 'entities.dict')                  #get the en2id/id2en
    re2id = get_id(datapath = 'relations.dict')                 #get the re2id/id2re
    enti_num, triple = get_num(en2id, re2id)                    #num of en/re, all triples in train and valid (saved with id). 
    data_path = save_triple(triple, "amie_data.tsv")            #dataset/FB15K237/relation_classify/amie_datapy.tsv
    print(data_path)
    target_dir = AMIE(data_path, minhc=HC, minpca=PCA, maxad=4) #dataset/FB15K237/relation_classify/minhc_0.5_minpca_0.8_maxad_4
    
    set_rela_symmetric,   \
    set_rela_inverse,     \
    set_rela_multiple, \
    set_rela_compose2,    \
    set_rela_compose3 = classify_relation(target_dir)
    # print(len(set_rela_symmetric))
    # print(set_rela_symmetric)

    list_all = [set_rela_symmetric, set_rela_inverse, set_rela_multiple, set_rela_compose2,set_rela_compose3]
    similarity(list_all)

    sym_list   =  classify_triples(re2id, set_rela_symmetric)
    inv_list   =  classify_triples(re2id, set_rela_inverse)
    mul_list   =  classify_triples(re2id, set_rela_multiple)
    comp2_list =  classify_triples(re2id, set_rela_compose2)
    comp3_list =  classify_triples(re2id, set_rela_compose3)

    entity_frequency(target_dir, "symmetric"  ,sym_list   , en2id, enti_num)
    entity_frequency(target_dir, "inverse"    ,inv_list   , en2id, enti_num)
    entity_frequency(target_dir, "multiple"   ,mul_list   , en2id, enti_num)
    entity_frequency(target_dir, "compose2"   ,comp2_list , en2id, enti_num)
    entity_frequency(target_dir, "compose3"   ,comp3_list , en2id, enti_num)

if __name__ == "__main__":
    dataset, PCA, HC= sys.argv[1:4]
    # print(sys.argv)
    if dataset[0]=='f' or 'F':
        relative_path = "dataset/FB15K237"
    else:
        relative_path = "dataset/WN18RR"
    
    main(PCA=float(PCA), HC=float(HC))