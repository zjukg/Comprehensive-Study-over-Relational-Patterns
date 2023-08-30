import numpy as np
from torch.utils.data import Dataset
import torch
import os
from collections import defaultdict as ddict
from IPython import embed
import re


class KGData(object):
    """Data preprocessing of kg data.

    Attributes:
        args: Some pre-set parameters, such as dataset path, etc. 
        ent2id: Encoding the entity in triples, type: dict.
        rel2id: Encoding the relation in triples, type: dict.
        id2ent: Decoding the entity in triples, type: dict.
        id2rel: Decoding the realtion in triples, type: dict.
        train_triples: Record the triples for training, type: list.
        valid_triples: Record the triples for validation, type: list.
        test_triples: Record the triples for testing, type: list.
        all_true_triples: Record all triples including train,valid and test, type: list.
        TrainTriples
        Relation2Tuple
        RelSub2Obj
        hr2t_train: Record the tail corresponding to the same head and relation, type: defaultdict(class:set).
        rt2h_train: Record the head corresponding to the same tail and relation, type: defaultdict(class:set).
        h2rt_train: Record the tail, relation corresponding to the same head, type: defaultdict(class:set).
        t2rh_train: Record the head, realtion corresponding to the same tail, type: defaultdict(class:set).
    """

    # TODO:把里面的函数再分一分，最基础的部分再初始化的使用调用，其他函数具体情况再调用
    def __init__(self, args):
        self.args = args

        #  基础部分
        self.ent2id = {}
        self.rel2id = {}
        # predictor需要
        self.id2ent = {}
        self.id2rel = {}
        # 存放三元组的id
        self.train_triples = []
        self.valid_triples = []
        self.test_triples = []
        self.all_true_triples = set()
        #  grounding 使用
        self.TrainTriples = {}
        self.Relation2Tuple = {}
        self.RelSub2Obj = {}

        self.hr2t_train = ddict(set)
        self.rt2h_train = ddict(set)
        self.h2rt_train = ddict(set)
        self.t2rh_train = ddict(set)
        self.get_id()
        self.get_triples_id()
        if args.use_weight:
            self.count = self.count_frequency(self.train_triples)
        
        #  relation pattern
        #  symmetric
        if args.use_sym_weight:
            self.sym_weight = self.sym_pattern()
        
        #  inverse
        if args.use_inv_weight:
            #  weight, rule, the number of this pattern that relation has, the max of the number
            self.inv_weight, self.inv_rule, self.inv_count, self.max_inv = self.inv_pattern()

        #  subrelation
        if args.use_sub_weight:
            self.sub_weight, self.sub_rule, self.sub_count, self.max_sub = self.sub_pattern()
        
        #  compose with 2 triples
        if args.use_comp2_weight:
            self.comp2_weight, self.comp2_rule, self.comp2_count, self.max_comp2, self.comp2_rel_inv = self.comp2_pattern()
        
        #  compose with 3 triples
        if args.use_comp3_weight:
            self.comp3_weight, self.comp3_rule, self.comp3_count, self.max_comp3 = self.comp3_pattern()

    def get_id(self):
        """Get entity/relation id, and entity/relation number.

        Update:
            self.ent2id: Entity to id.
            self.rel2id: Relation to id.
            self.id2ent: id to Entity.
            self.id2rel: id to Relation.
            self.args.num_ent: Entity number.
            self.args.num_rel: Relation number.
        """
        with open(os.path.join(self.args.data_path, "entities.dict")) as fin:
            for line in fin:
                eid, entity = line.strip().split("\t")
                self.ent2id[entity] = int(eid)
                self.id2ent[int(eid)] = entity
    
        with open(os.path.join(self.args.data_path, "relations.dict")) as fin:
            for line in fin:
                rid, relation = line.strip().split("\t")
                self.rel2id[relation] = int(rid)
                self.id2rel[int(rid)] = relation

        self.args.num_ent = len(self.ent2id)
        self.args.num_rel = len(self.rel2id)

    def get_triples_id(self):
        """Get triples id, save in the format of (h, r, t).

        Update:
            self.train_triples: Train dataset triples id.
            self.valid_triples: Valid dataset triples id.
            self.test_triples: Test dataset triples id.
        """
        
        with open(os.path.join(self.args.data_path, "train.txt")) as f:
            for line in f.readlines():
                h, r, t = line.strip().split()
                self.train_triples.append(
                    (self.ent2id[h], self.rel2id[r], self.ent2id[t])
                )
                
                tmp = str(self.ent2id[h]) + '\t' + str(self.rel2id[r]) + '\t' + str(self.ent2id[t])
                self.TrainTriples[tmp] = True

                iRelationID = self.rel2id[r]
                strValue = str(h) + "#" + str(t)
                if not iRelationID in self.Relation2Tuple:
                    tmpLst = []
                    tmpLst.append(strValue)
                    self.Relation2Tuple[iRelationID] = tmpLst
                else:
                    self.Relation2Tuple[iRelationID].append(strValue)

                iRelationID = self.rel2id[r]
                iSubjectID = self.ent2id[h]
                iObjectID = self.ent2id[t]
                tmpMap = {}
                tmpMap_in = {}
                if not iRelationID in self.RelSub2Obj:
                    if not iSubjectID in tmpMap:
                        tmpMap_in.clear()
                        tmpMap_in[iObjectID] = True
                        tmpMap[iSubjectID] = tmpMap_in
                    else:
                        tmpMap[iSubjectID][iObjectID] = True
                    self.RelSub2Obj[iRelationID] = tmpMap
                else:
                    tmpMap = self.RelSub2Obj[iRelationID]
                    if not iSubjectID in tmpMap:
                        tmpMap_in.clear()
                        tmpMap_in[iObjectID] = True
                        tmpMap[iSubjectID] = tmpMap_in
                    else:
                        tmpMap[iSubjectID][iObjectID] = True
                    self.RelSub2Obj[iRelationID] = tmpMap  # 是不是应该要加？

        with open(os.path.join(self.args.data_path, "valid.txt")) as f:
            for line in f.readlines():
                h, r, t = line.strip().split()
                self.valid_triples.append(
                    (self.ent2id[h], self.rel2id[r], self.ent2id[t])
                )

        with open(os.path.join(self.args.data_path, "test.txt")) as f:
            for line in f.readlines():
                h, r, t = line.strip().split()
                self.test_triples.append(
                    (self.ent2id[h], self.rel2id[r], self.ent2id[t])
                )

        self.all_true_triples = set(
            self.train_triples + self.valid_triples + self.test_triples
        )

    def get_hr2t_rt2h_from_train(self):
        """Get the set of hr2t and rt2h from train dataset, the data type is numpy.

        Update:
            self.hr2t_train: The set of hr2t.
            self.rt2h_train: The set of rt2h.
        """
        
        for h, r, t in self.train_triples:
            self.hr2t_train[(h, r)].add(t)
            self.rt2h_train[(r, t)].add(h)
        for h, r in self.hr2t_train:
            self.hr2t_train[(h, r)] = np.array(list(self.hr2t_train[(h, r)]))
        for r, t in self.rt2h_train:
            self.rt2h_train[(r, t)] = np.array(list(self.rt2h_train[(r, t)]))

    @staticmethod
    def count_frequency(triples, start=4):
        '''Get frequency of a partial triple like (head, relation) or (relation, tail).
        
        The frequency will be used for subsampling like word2vec.
        
        Args:
            triples: Sampled triples.
            start: Initial count number.

        Returns:
            count: Record the number of (head, relation).
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation-1) not in count:
                count[(tail, -relation-1)] = start
            else:
                count[(tail, -relation-1)] += 1
        return count
        
    def sym_pattern(self):
        print("sym_pattern", self.args.lambda_sym)
        rule_mining_path = os.path.join(self.args.data_path,"relation_classify","minhc_"+str(self.args.minhc)+"_minpca_"+str(self.args.minpca)+"_maxad_"+str(self.args.maxad)) #读文件路径
        target_path = os.path.join( rule_mining_path, "result.txt") #存文件路径
        sym_weight = []
        for i in range(len(self.rel2id)): sym_weight.append([0.0,0.0])
        with open(target_path) as fin:
            for line in fin:
                if(line[0]!='?'):
                    continue
                else:
                    ls = re.findall(r"(\d+\.?\d*|\?[a-z])",line)
                    if len(ls) == 13:  # r1->r2
                        h_1, r_1, t_1, h_2, r_2, t_2 = ls[:6]
                        #   symmetric
                        if r_1==r_2 and h_1==t_2 and h_2==t_1:
                            sym_weight[int(r_2)][0] = float(ls[6])
                            sym_weight[int(r_2)][1] = float(ls[8])
        return sym_weight
    
    def inv_pattern(self):
        print("inv_pattern", self.args.lambda_inv)
        rule_mining_path = os.path.join(self.args.data_path,"relation_classify","minhc_"+str(self.args.minhc)+"_minpca_"+str(self.args.minpca)+"_maxad_"+str(self.args.maxad)) #读文件路径
        target_path = os.path.join( rule_mining_path, "result.txt") #存文件路径
        print(target_path)
        max_inv = [0 for i in range(len(self.rel2id))]
        with open(target_path) as fin:
            for line in fin:
                if(line[0]!='?'):
                    continue
                else:
                    ls = re.findall(r"(\d+\.?\d*|\?[a-z])",line)
                    if len(ls) == 13:  # r1->r2
                        h_1, r_1, t_1, h_2, r_2, t_2 = ls[:6]
                        #   inverse
                        if r_1!=r_2 and h_1==t_2 and h_2==t_1:
                            # set_rela_symmetric.add(r_2)
                            max_inv[int(r_2)] += 1
        max_length = max(max_inv)
        print("length of pattern: ",max_length)
        inv_weight = [ [ [ 0 for k in range(2) ]for i in range(max_length)] for j in range(len(self.rel2id))]
        inv_rule = [[0 for i in range(max_length)] for j in range(len(self.rel2id))]
        inv_count = [0 for i in range(len(self.rel2id))]
        with open(target_path) as fin:
            for line in fin:
                if(line[0]!='?'):
                    continue
                else:
                    ls = re.findall(r"(\d+\.?\d*|\?[a-z])",line)
                    if len(ls) == 13:  # r1->r2
                        h_1, r_1, t_1, h_2, r_2, t_2 = ls[:6]
                        #   inverse
                        if r_1!=r_2 and h_1==t_2 and h_2==t_1:
                            inv_weight[int(r_2)][inv_count[int(r_2)]][0] = float(ls[6])
                            inv_weight[int(r_2)][inv_count[int(r_2)]][1] = float(ls[8])
                            inv_rule[int(r_2)][inv_count[int(r_2)]] = int(r_1)
                            inv_count[int(r_2)] += 1
        return inv_weight, inv_rule, inv_count, max_length

    def sub_pattern(self):
        print("sub_pattern", self.args.lambda_sub)
        rule_mining_path = os.path.join(self.args.data_path,"relation_classify","minhc_"+str(self.args.minhc)+"_minpca_"+str(self.args.minpca)+"_maxad_"+str(self.args.maxad)) #读文件路径
        target_path = os.path.join( rule_mining_path, "result.txt") #存文件路径
        max_sub = [0 for i in range(len(self.rel2id))]
        with open(target_path) as fin:
            for line in fin:
                if(line[0]!='?'):
                    continue
                else:
                    ls = re.findall(r"(\d+\.?\d*|\?[a-z])",line)
                    if len(ls) == 13:  # r1->r2
                        h_1, r_1, t_1, h_2, r_2, t_2 = ls[:6]
                        #   subrelation
                        if r_1!=r_2 and h_1==h_2 and t_1==t_2:
                            max_sub[int(r_2)] += 1
        max_length = max(max_sub)
        print("length of pattern: ",max_length)
        sub_weight = [ [ [ 0 for k in range(2) ]for i in range(max_length)] for j in range(len(self.rel2id))]
        sub_rule = [[0 for i in range(max_length)] for j in range(len(self.rel2id))]
        sub_count = [0 for i in range(len(self.rel2id))]
        with open(target_path) as fin:
            for line in fin:
                if(line[0]!='?'):
                    continue
                else:
                    ls = re.findall(r"(\d+\.?\d*|\?[a-z])",line)
                    if len(ls) == 13:  # r1->r2
                        h_1, r_1, t_1, h_2, r_2, t_2 = ls[:6]
                        #   subrelation
                        if r_1!=r_2 and h_1==h_2 and t_1==t_2:
                            sub_weight[int(r_2)][sub_count[int(r_2)]][0] = float(ls[6])
                            sub_weight[int(r_2)][sub_count[int(r_2)]][1] = float(ls[8])
                            sub_rule[int(r_2)][sub_count[int(r_2)]] = int(r_1)
                            sub_count[int(r_2)] += 1
        return sub_weight, sub_rule, sub_count, max_length
    
    def comp2_pattern(self):
        print("comp2_pattern", self.args.lambda_comp2)
        rule_mining_path = os.path.join(self.args.data_path,"relation_classify","minhc_"+str(self.args.minhc)+"_minpca_"+str(self.args.minpca)+"_maxad_"+str(self.args.maxad)) #读文件路径
        target_path = os.path.join( rule_mining_path, "result.txt") #存文件路径
        max_comp2 = [0 for i in range(len(self.rel2id))]
        with open(target_path) as fin:
            for line in fin:
                if(line[0]!='?'):
                    continue
                else:
                    ls = re.findall(r"(\d+\.?\d*|\?[a-z])",line)
                    if len(ls) == 16:  # r1+r2->r3
                        h_1, r_1, t_1, h_2, r_2, t_2, h_3, r_3, t_3 = ls[:9]
                        #   comp2relation
                        max_comp2[int(r_3)] += 1
        max_length = max(max_comp2)
        print("length of pattern: ",max_length)
        comp2_weight = [ [ [ 0 for k in range(2) ]for i in range(max_length)] for j in range(len(self.rel2id))]
        comp2_rule   = [ [ [ 0 for k in range(2) ]for i in range(max_length)] for j in range(len(self.rel2id))]
        comp2_rel_inv= [ [ [ 0 for k in range(2) ]for i in range(max_length)] for j in range(len(self.rel2id))]
        comp2_count  = [ 0 for i in range(len(self.rel2id))]
        with open(target_path) as fin:
            for line in fin:
                if(line[0]!='?'):
                    continue
                else:
                    ls = re.findall(r"(\d+\.?\d*|\?[a-z])",line)
                    if len(ls) == 16:  # r1+r2->r3
                        h_1, r_1, t_1, h_2, r_2, t_2, h_3, r_3, t_3 = ls[:9]
                        #   comp2relation
                        if h_1==h_3 and t_1==h_2 and t_2==t_3 :   # r1 r2
                            comp2_weight[int(r_3)][comp2_count[int(r_3)]][0] = float(ls[9])
                            comp2_weight[int(r_3)][comp2_count[int(r_3)]][1] = float(ls[11])
                            comp2_rule[int(r_3)][comp2_count[int(r_3)]][0] = int(r_1)
                            comp2_rule[int(r_3)][comp2_count[int(r_3)]][1] = int(r_2)

                            comp2_rel_inv[int(r_3)][comp2_count[int(r_3)]][0] = 0
                            comp2_rel_inv[int(r_3)][comp2_count[int(r_3)]][1] = 0
                            comp2_count[int(r_3)] += 1
                        
                        elif h_1==t_2 and t_1==t_3 and h_2==h_3 : # r2 r1
                            comp2_weight[int(r_3)][comp2_count[int(r_3)]][0] = float(ls[9])
                            comp2_weight[int(r_3)][comp2_count[int(r_3)]][1] = float(ls[11])
                            comp2_rule[int(r_3)][comp2_count[int(r_3)]][0] = int(r_2)
                            comp2_rule[int(r_3)][comp2_count[int(r_3)]][1] = int(r_1)

                            comp2_rel_inv[int(r_3)][comp2_count[int(r_3)]][0] = 0
                            comp2_rel_inv[int(r_3)][comp2_count[int(r_3)]][1] = 0
                            comp2_count[int(r_3)] += 1

                        elif h_1==h_2 and t_1==t_3 and t_2==h_3 : #r2-1 r1
                            comp2_weight[int(r_3)][comp2_count[int(r_3)]][0] = float(ls[9])
                            comp2_weight[int(r_3)][comp2_count[int(r_3)]][1] = float(ls[11])
                            comp2_rule[int(r_3)][comp2_count[int(r_3)]][0] = int(r_2)
                            comp2_rule[int(r_3)][comp2_count[int(r_3)]][1] = int(r_1)

                            comp2_rel_inv[int(r_3)][comp2_count[int(r_3)]][0] = 1
                            comp2_rel_inv[int(r_3)][comp2_count[int(r_3)]][1] = 0
                            comp2_count[int(r_3)] += 1

                        elif h_1==h_2 and t_1==h_3 and t_2==t_3 : #r1-1 r2
                            comp2_weight[int(r_3)][comp2_count[int(r_3)]][0] = float(ls[9])
                            comp2_weight[int(r_3)][comp2_count[int(r_3)]][1] = float(ls[11])
                            comp2_rule[int(r_3)][comp2_count[int(r_3)]][0] = int(r_1)
                            comp2_rule[int(r_3)][comp2_count[int(r_3)]][1] = int(r_2)

                            comp2_rel_inv[int(r_3)][comp2_count[int(r_3)]][0] = 1
                            comp2_rel_inv[int(r_3)][comp2_count[int(r_3)]][1] = 0
                            comp2_count[int(r_3)] += 1

                        elif h_1==h_3 and t_1==t_2 and h_2==t_3 : #r1 r2-1
                            comp2_weight[int(r_3)][comp2_count[int(r_3)]][0] = float(ls[9])
                            comp2_weight[int(r_3)][comp2_count[int(r_3)]][1] = float(ls[11])
                            comp2_rule[int(r_3)][comp2_count[int(r_3)]][0] = int(r_1)
                            comp2_rule[int(r_3)][comp2_count[int(r_3)]][1] = int(r_2)

                            comp2_rel_inv[int(r_3)][comp2_count[int(r_3)]][0] = 0
                            comp2_rel_inv[int(r_3)][comp2_count[int(r_3)]][1] = 1
                            comp2_count[int(r_3)] += 1

                        elif h_1==t_3 and t_1==t_2 and h_2==h_3 : #r2 r1-1
                            comp2_weight[int(r_3)][comp2_count[int(r_3)]][0] = float(ls[9])
                            comp2_weight[int(r_3)][comp2_count[int(r_3)]][1] = float(ls[11])
                            comp2_rule[int(r_3)][comp2_count[int(r_3)]][0] = int(r_2)
                            comp2_rule[int(r_3)][comp2_count[int(r_3)]][1] = int(r_1)

                            comp2_rel_inv[int(r_3)][comp2_count[int(r_3)]][0] = 0
                            comp2_rel_inv[int(r_3)][comp2_count[int(r_3)]][1] = 1
                            comp2_count[int(r_3)] += 1


                        elif h_1==t_2 and t_1==h_3 and h_2==t_3 : #r1-1 r2-1
                            comp2_weight[int(r_3)][comp2_count[int(r_3)]][0] = float(ls[9])
                            comp2_weight[int(r_3)][comp2_count[int(r_3)]][1] = float(ls[11])
                            comp2_rule[int(r_3)][comp2_count[int(r_3)]][0] = int(r_1)
                            comp2_rule[int(r_3)][comp2_count[int(r_3)]][1] = int(r_2)

                            comp2_rel_inv[int(r_3)][comp2_count[int(r_3)]][0] = 1
                            comp2_rel_inv[int(r_3)][comp2_count[int(r_3)]][1] = 1
                            comp2_count[int(r_3)] += 1

                        else:                                     #r2-1 r1-1
                            comp2_weight[int(r_3)][comp2_count[int(r_3)]][0] = float(ls[9])
                            comp2_weight[int(r_3)][comp2_count[int(r_3)]][1] = float(ls[11])
                            comp2_rule[int(r_3)][comp2_count[int(r_3)]][0] = int(r_2)
                            comp2_rule[int(r_3)][comp2_count[int(r_3)]][1] = int(r_1)

                            comp2_rel_inv[int(r_3)][comp2_count[int(r_3)]][0] = 1
                            comp2_rel_inv[int(r_3)][comp2_count[int(r_3)]][1] = 1
                            comp2_count[int(r_3)] += 1


                        
        return comp2_weight, comp2_rule, comp2_count, max_length, comp2_rel_inv

    def comp3_pattern(self):
        print("comp3_pattern", self.args.lambda_comp3)
        rule_mining_path = os.path.join(self.args.data_path,"relation_classify","minhc_"+str(self.args.minhc)+"_minpca_"+str(self.args.minpca)+"_maxad_"+str(self.args.maxad)) #读文件路径
        target_path = os.path.join( rule_mining_path, "result.txt") #存文件路径
        max_comp3 = [0 for i in range(len(self.rel2id))]
        with open(target_path) as fin:
            for line in fin:
                if(line[0]!='?'):
                    continue
                else:
                    ls = re.findall(r"(\d+\.?\d*|\?[a-z])",line)
                    if len(ls) == 19:  # r1+r2+r3->r4
                        h_1, r_1, t_1, h_2, r_2, t_2, h_3, r_3, t_3, h_4, r_4, h_4 = ls[:12]
                        #   comp3relation
                        if ( h_1==h_4 and t_1==h_2 and t_2==h_3 and t_3==h_4) or \
                           ( h_1==h_4 and t_1==h_3 and t_3==h_2 and t_2==h_4) or \
                           ( h_2==h_4 and t_2==h_1 and t_1==h_3 and t_3==h_4) or \
                           ( h_2==h_4 and t_2==h_3 and t_3==h_1 and t_1==h_4) or \
                           ( h_3==h_4 and t_3==h_1 and t_1==h_2 and t_2==h_4) or \
                           ( h_3==h_4 and t_3==h_2 and t_2==h_1 and t_1==h_4):
                            max_comp3[int(r_4)] += 1
        max_length = max(max_comp3)
        print("length of pattern: ",max_length)
        comp3_weight = [ [ [ 0 for k in range(2) ]for i in range(max_length)] for j in range(len(self.rel2id))]
        comp3_rule   = [ [ [ 0 for k in range(3) ]for i in range(max_length)] for j in range(len(self.rel2id))]
        comp3_count  = [ 0 for i in range(len(self.rel2id))]
        with open(target_path) as fin:
            for line in fin:
                if(line[0]!='?'):
                    continue
                else:
                    ls = re.findall(r"(\d+\.?\d*|\?[a-z])",line)
                    if len(ls) == 19:  # r1+r2+r3->r4
                        h_1, r_1, t_1, h_2, r_2, t_2, h_3, r_3, t_3, h_4, r_4, h_4 = ls[:12]
                        #   comp3relation
                        if ( h_1==h_4 and t_1==h_2 and t_2==h_3 and t_3==h_4) or \
                           ( h_1==h_4 and t_1==h_3 and t_3==h_2 and t_2==h_4) or \
                           ( h_2==h_4 and t_2==h_1 and t_1==h_3 and t_3==h_4) or \
                           ( h_2==h_4 and t_2==h_3 and t_3==h_1 and t_1==h_4) or \
                           ( h_3==h_4 and t_3==h_1 and t_1==h_2 and t_2==h_4) or \
                           ( h_3==h_4 and t_3==h_2 and t_2==h_1 and t_1==h_4):
                            comp3_weight[int(r_3)][comp3_count[int(r_3)]][0] = float(ls[12])
                            comp3_weight[int(r_3)][comp3_count[int(r_3)]][1] = float(ls[14])
                            comp3_rule[int(r_3)][comp3_count[int(r_3)]][0] = int(r_1)
                            comp3_rule[int(r_3)][comp3_count[int(r_3)]][1] = int(r_2)
                            comp3_rule[int(r_3)][comp3_count[int(r_3)]][2] = int(r_3)
                            comp3_count[int(r_3)] += 1
                        
        return comp3_weight, comp3_rule, comp3_count, max_length
    
    def get_h2rt_t2hr_from_train(self):
        """Get the set of h2rt and t2hr from train dataset, the data type is numpy.

        Update:
            self.h2rt_train: The set of h2rt.
            self.t2rh_train: The set of t2hr.
        """
        for h, r, t in self.train_triples:
            self.h2rt_train[h].add((r, t))
            self.t2rh_train[t].add((r, h))
        for h in self.h2rt_train:
            self.h2rt_train[h] = np.array(list(self.h2rt_train[h]))
        for t in self.t2rh_train:
            self.t2rh_train[t] = np.array(list(self.t2rh_train[t]))
        
    def get_hr_trian(self):
        '''Change the generation mode of batch.
        Merging triples which have same head and relation for 1vsN training mode.

        Returns:
            self.train_triples: The tuple(hr, t) list for training
        '''
        self.t_triples = self.train_triples 
        self.train_triples = [ (hr, list(t)) for (hr,t) in self.hr2t_train.items()]

class BaseSampler(KGData):
    """Traditional random sampling mode.
    """
    def __init__(self, args):
        super().__init__(args)
        self.get_hr2t_rt2h_from_train()

    def corrupt_head(self, t, r, num_max=1):
        """Negative sampling of head entities.

        Args:
            t: Tail entity in triple.
            r: Relation in triple.
            num_max: The maximum of negative samples generated 

        Returns:
            neg: The negative sample of head entity filtering out the positive head entity.
        """
        tmp = torch.randint(low=0, high=self.args.num_ent, size=(num_max,)).numpy()
        if not self.args.filter_flag:
            return tmp
        mask = np.in1d(tmp, self.rt2h_train[(r, t)], assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg

    def corrupt_tail(self, h, r, num_max=1):
        """Negative sampling of tail entities.

        Args:
            h: Head entity in triple.
            r: Relation in triple.
            num_max: The maximum of negative samples generated 

        Returns:
            neg: The negative sample of tail entity filtering out the positive tail entity.
        """
        tmp = torch.randint(low=0, high=self.args.num_ent, size=(num_max,)).numpy()
        if not self.args.filter_flag:
            return tmp
        mask = np.in1d(tmp, self.hr2t_train[(h, r)], assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg

    def head_batch(self, h, r, t, neg_size=None):
        """Negative sampling of head entities.

        Args:
            h: Head entity in triple
            t: Tail entity in triple.
            r: Relation in triple.
            neg_size: The size of negative samples.

        Returns:
            The negative sample of head entity. [neg_size]
        """
        neg_list = []
        neg_cur_size = 0
        while neg_cur_size < neg_size:
            neg_tmp = self.corrupt_head(t, r, num_max=(neg_size - neg_cur_size) * 2)
            neg_list.append(neg_tmp)
            neg_cur_size += len(neg_tmp)
        return np.concatenate(neg_list)[:neg_size]

    def tail_batch(self, h, r, t, neg_size=None):
        """Negative sampling of tail entities.

        Args:
            h: Head entity in triple
            t: Tail entity in triple.
            r: Relation in triple.
            neg_size: The size of negative samples.

        Returns:
            The negative sample of tail entity. [neg_size]
        """
        neg_list = []
        neg_cur_size = 0
        while neg_cur_size < neg_size:
            neg_tmp = self.corrupt_tail(h, r, num_max=(neg_size - neg_cur_size) * 2)
            neg_list.append(neg_tmp)
            neg_cur_size += len(neg_tmp)
        return np.concatenate(neg_list)[:neg_size]

    def get_train(self):
        return self.train_triples

    def get_valid(self):
        return self.valid_triples

    def get_test(self):
        return self.test_triples

    def get_all_true_triples(self):
        return self.all_true_triples


class RevSampler(KGData):
    """Adding reverse triples in traditional random sampling mode.

    For each triple (h, r, t), generate the reverse triple (t, r`, h).
    r` = r + num_rel.

    Attributes:
        hr2t_train: Record the tail corresponding to the same head and relation, type: defaultdict(class:set).
        rt2h_train: Record the head corresponding to the same tail and relation, type: defaultdict(class:set).
    """
    def __init__(self, args):
        super().__init__(args)
        self.hr2t_train = ddict(set)
        self.rt2h_train = ddict(set)
        self.add_reverse_relation()
        self.add_reverse_triples()
        self.get_hr2t_rt2h_from_train()

    def add_reverse_relation(self):
        """Get entity/relation/reverse relation id, and entity/relation number.

        Update:
            self.ent2id: Entity id.
            self.rel2id: Relation id.
            self.args.num_ent: Entity number.
            self.args.num_rel: Relation number.
        """
        
        with open(os.path.join(self.args.data_path, "relations.dict")) as fin:
            len_rel2id = len(self.rel2id)
            for line in fin:
                rid, relation = line.strip().split("\t")
                self.rel2id[relation + "_reverse"] = int(rid) + len_rel2id
                self.id2rel[int(rid) + len_rel2id] = relation + "_reverse"
        self.args.num_rel = len(self.rel2id)

    def add_reverse_triples(self):
        """Generate reverse triples (t, r`, h).

        Update:
            self.train_triples: Triples for training.
            self.valid_triples: Triples for validation.
            self.test_triples: Triples for testing.
            self.all_ture_triples: All triples including train, valid and test.
        """

        with open(os.path.join(self.args.data_path, "train.txt")) as f:
            for line in f.readlines():
                h, r, t = line.strip().split()
                self.train_triples.append(
                    (self.ent2id[t], self.rel2id[r + "_reverse"], self.ent2id[h])
                )

        with open(os.path.join(self.args.data_path, "valid.txt")) as f:
            for line in f.readlines():
                h, r, t = line.strip().split()
                self.valid_triples.append(
                    (self.ent2id[t], self.rel2id[r + "_reverse"], self.ent2id[h])
                )

        with open(os.path.join(self.args.data_path, "test.txt")) as f:
            for line in f.readlines():
                h, r, t = line.strip().split()
                self.test_triples.append(
                    (self.ent2id[t], self.rel2id[r + "_reverse"], self.ent2id[h])
                )

        self.all_true_triples = set(
            self.train_triples + self.valid_triples + self.test_triples
        )

    def get_train(self):
        return self.train_triples

    def get_valid(self):
        return self.valid_triples

    def get_test(self):
        return self.test_triples

    def get_all_true_triples(self):
        return self.all_true_triples    
    
    def corrupt_head(self, t, r, num_max=1):
        """Negative sampling of head entities.

        Args:
            t: Tail entity in triple.
            r: Relation in triple.
            num_max: The maximum of negative samples generated 

        Returns:
            neg: The negative sample of head entity filtering out the positive head entity.
        """
        tmp = torch.randint(low=0, high=self.args.num_ent, size=(num_max,)).numpy()
        if not self.args.filter_flag:
            return tmp
        mask = np.in1d(tmp, self.rt2h_train[(r, t)], assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg

    def corrupt_tail(self, h, r, num_max=1):
        """Negative sampling of tail entities.

        Args:
            h: Head entity in triple.
            r: Relation in triple.
            num_max: The maximum of negative samples generated 

        Returns:
            neg: The negative sample of tail entity filtering out the positive tail entity.
        """
        tmp = torch.randint(low=0, high=self.args.num_ent, size=(num_max,)).numpy()
        if not self.args.filter_flag:
            return tmp
        mask = np.in1d(tmp, self.hr2t_train[(h, r)], assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg

    def head_batch(self, h, r, t, neg_size=None):
        """Negative sampling of head entities.

        Args:
            h: Head entity in triple
            t: Tail entity in triple.
            r: Relation in triple.
            neg_size: The size of negative samples.

        Returns:
            The negative sample of head entity. [neg_size]
        """
        neg_list = []
        neg_cur_size = 0
        while neg_cur_size < neg_size:
            neg_tmp = self.corrupt_head(t, r, num_max=(neg_size - neg_cur_size) * 2)
            neg_list.append(neg_tmp)
            neg_cur_size += len(neg_tmp)
        return np.concatenate(neg_list)[:neg_size]

    def tail_batch(self, h, r, t, neg_size=None):
        """Negative sampling of tail entities.

        Args:
            h: Head entity in triple
            t: Tail entity in triple.
            r: Relation in triple.
            neg_size: The size of negative samples.

        Returns:
            The negative sample of tail entity. [neg_size]
        """
        neg_list = []
        neg_cur_size = 0
        while neg_cur_size < neg_size:
            neg_tmp = self.corrupt_tail(h, r, num_max=(neg_size - neg_cur_size) * 2)
            neg_list.append(neg_tmp)
            neg_cur_size += len(neg_tmp)
        return np.concatenate(neg_list)[:neg_size]