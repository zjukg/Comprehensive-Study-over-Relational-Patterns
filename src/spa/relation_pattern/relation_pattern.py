import torch.nn as nn
import torch
from IPython import embed
from .model import Model


class RP(Model):
    def __init__(self, args):
        super(RP, self).__init__()
        self.args = args

    def func_sym(self, batch, mode):
        triples = batch["symmetric_sample"] #[bs,3]
        symmode = "head_predict" if mode == "tail_predict" else "tail_predict"

        head_emb, relation_emb, tail_emb = self.tri2emb(triples, mode=symmode)
        sym_score = self.score_func(head_emb, relation_emb, tail_emb, symmode)

        sym_weight = batch["sym_weight"]
        JC_sym = (( sym_weight[:,0] + sym_weight[:,1] ) / 2).unsqueeze(1)

        return sym_score, JC_sym
    
    def func_inv(self, batch, mode):
        triples = batch["inverse_sample"] #[bs,max_inv,3]
        inv_weight = batch["inv_weight"]
        inv_score_list = []

        for idx, triple in enumerate(torch.chunk(triples, batch["max_inv"], dim = 1)):
            triple = torch.squeeze(triple) #[bs,3]
            invmode = "head_predict" if mode == "tail_predict" else "tail_predict"

            head_emb, relation_emb, tail_emb = self.tri2emb(triple, mode=invmode)
            inv_score_tmp = self.score_func(head_emb, relation_emb, tail_emb, invmode)

            JC_inv = (( inv_weight[:,idx,0] + inv_weight[:,idx,1] ) / 2).unsqueeze(1)
            inv_score_list.append(JC_inv*inv_score_tmp)
        
        inv_score = sum(inv_score for inv_score in inv_score_list)
        avg_JC_inv = torch.mean(inv_weight,dim=-1)
        
        ls_avg_JC = [] #去除0后的取平均
        for tensor in avg_JC_inv: 
            tmp_list = []
            for idx,e in enumerate(tensor):
                if idx == 0 and e == 0 :tmp_list.append(e);break
                elif e == 0: break
                else: tmp_list.append(e)
            ls_avg_JC.append(sum(tmp_list)/len(tmp_list))
        
        avg_JC_inv = torch.tensor(ls_avg_JC).to(torch.device('cuda')).unsqueeze(1)
        sum_JC_inv = torch.sum(torch.mean(inv_weight,dim=-1),dim=-1).unsqueeze(1)
        sum_JC_inv = torch.where(sum_JC_inv<0.01, torch.tensor([1.]).to(torch.device('cuda')),sum_JC_inv) #将所有0替换为1 防止/0=nan

        return inv_score/sum_JC_inv, avg_JC_inv
    
    def func_sub(self, batch, mode):
        triples = batch["subrelation_sample"] #[bs,max_sub,3]
        sub_weight = batch["sub_weight"]
        sub_score_list = []

        for idx, triple in enumerate(torch.chunk(triples, batch["max_sub"], dim = 1)):
            triple = torch.squeeze(triple) #[bs,3]
            submode = mode

            head_emb, relation_emb, tail_emb = self.tri2emb(triple, mode=submode)
            sub_score_tmp = self.score_func(head_emb, relation_emb, tail_emb, submode)

            JC_sub = (( sub_weight[:,idx,0] + sub_weight[:,idx,1] ) / 2).unsqueeze(1)
            sub_score_list.append(JC_sub*sub_score_tmp)
        
        sub_score = sum(sub_score for sub_score in sub_score_list)
        avg_JC_sub = torch.mean(sub_weight,dim=-1)
        
        ls_avg_JC = [] #去除0后的取平均
        for tensor in avg_JC_sub: 
            tmp_list = []
            for idx,e in enumerate(tensor):
                if idx == 0 and e == 0 :tmp_list.append(e);break
                elif e == 0: break
                else: tmp_list.append(e)
            ls_avg_JC.append(sum(tmp_list)/len(tmp_list))
        
        avg_JC_sub = torch.tensor(ls_avg_JC).to(torch.device('cuda')).unsqueeze(1)
        sum_JC_sub = torch.sum(torch.mean(sub_weight,dim=-1),dim=-1).unsqueeze(1)

        sum_JC_sub = torch.where(sum_JC_sub<0.01, torch.tensor([1.]).to(torch.device('cuda')),sum_JC_sub) #将所有0替换为1 防止/0=nan

        return sub_score/sum_JC_sub, avg_JC_sub
    
    def func_comp2(self, batch, mode):
        # import pdb;pdb.set_trace()
        triples1 = batch["comp2_sample1"] #[bs,max_comp2,3]
        triples2 = batch["comp2_sample2"] #[bs,max_comp2,3]
        comp2_weight = batch["comp2_weight"]
        comp2_score_list = []

        tmp_triples1 = torch.chunk(triples1, batch["max_comp2"], dim = 1)
        tmp_triples2 = torch.chunk(triples2, batch["max_comp2"], dim = 1)

        for idx, triple in enumerate(tmp_triples1):
            triple1 = torch.squeeze(triple)                 #[bs,3]
            triple2 = torch.squeeze(tmp_triples2[idx])      #[bs,3]
            comp2mode = mode

            head_emb, relation_emb1, tail_emb = self.tri2emb(triple1, mode=comp2mode)
            head_emb, relation_emb2, tail_emb = self.tri2emb(triple2, mode=comp2mode)
            comp2_score_tmp = self.score_func_comp2(head_emb, relation_emb1, relation_emb2, tail_emb, comp2mode)
            
            JC_comp2 = (( comp2_weight[:,idx,0] + comp2_weight[:,idx,1] ) / 2).unsqueeze(1)
            comp2_score_list.append(JC_comp2*comp2_score_tmp)
        
        comp2_score = sum(comp2_score for comp2_score in comp2_score_list)
        avg_JC_comp2 = torch.mean(comp2_weight,dim=-1)
        
        ls_avg_JC = [] #去除0后的取平均
        for tensor in avg_JC_comp2: 
            tmp_list = []
            for idx,e in enumerate(tensor):
                if idx == 0 and e == 0 :tmp_list.append(e);break
                elif e == 0: break
                else: tmp_list.append(e)
            ls_avg_JC.append(sum(tmp_list)/len(tmp_list))
        
        avg_JC_comp2 = torch.tensor(ls_avg_JC).to(torch.device('cuda')).unsqueeze(1)
        sum_JC_comp2 = torch.sum(torch.mean(comp2_weight,dim=-1),dim=-1).unsqueeze(1)
        
        sum_JC_comp2 = torch.where(sum_JC_comp2<0.01, torch.tensor([1.]).to(torch.device('cuda')),sum_JC_comp2) #将所有0替换为1 防止/0=nan
        return comp2_score/sum_JC_comp2, avg_JC_comp2
    
    def func_comp3(self, batch, mode):
        # import pdb;pdb.set_trace()
        triples1 = batch["comp3_sample1"] #[bs,max_comp3,3]
        triples2 = batch["comp3_sample2"] #[bs,max_comp3,3]
        triples3 = batch["comp3_sample3"] #[bs,max_comp3,3]
        comp3_weight = batch["comp3_weight"]
        comp3_score_list = []

        tmp_triples1 = torch.chunk(triples1, batch["max_comp3"], dim = 1)
        tmp_triples2 = torch.chunk(triples2, batch["max_comp3"], dim = 1)
        tmp_triples3 = torch.chunk(triples3, batch["max_comp3"], dim = 1)

        for idx, triple in enumerate(tmp_triples1):

            triple1 = torch.squeeze(triple)                 #[bs,3]
            triple2 = torch.squeeze(tmp_triples2[idx])      #[bs,3]
            triple3 = torch.squeeze(tmp_triples3[idx])      #[bs,3]
            comp3mode = mode

            head_emb, relation_emb1, tail_emb = self.tri2emb(triple1, mode=comp3mode)
            head_emb, relation_emb2, tail_emb = self.tri2emb(triple2, mode=comp3mode)
            head_emb, relation_emb3, tail_emb = self.tri2emb(triple3, mode=comp3mode)

            
            relation_emb = relation_emb1 + relation_emb2 + relation_emb3
            comp3_score_tmp = self.score_func(head_emb, relation_emb, tail_emb, comp3mode)
            # import pdb;pdb.set_trace()
            lambda_comp3 = (( comp3_weight[:,idx,0] + comp3_weight[:,idx,1] ) / 2).unsqueeze(1)
            # import pdb;pdb.set_trace()

            comp3_score_list.append(lambda_comp3*comp3_score_tmp)
        
        comp3_score = sum(comp3_score for comp3_score in comp3_score_list)
        avg_lambda_comp3 = torch.mean(comp3_weight,dim=-1)
        
        ls_avg_lambda = [] #去除0后的取平均
        for tensor in avg_lambda_comp3: 
            tmp_list = []
            for idx,e in enumerate(tensor):
                if idx == 0 and e == 0 :tmp_list.append(e);break
                elif e == 0: break
                else: tmp_list.append(e)
            ls_avg_lambda.append(sum(tmp_list)/len(tmp_list))
        
        avg_lambda_comp3 = torch.tensor(ls_avg_lambda).to(torch.device('cuda')).unsqueeze(1)
        sum_lambda_comp3 = torch.sum(torch.mean(comp3_weight,dim=-1),dim=-1).unsqueeze(1)
        
        sum_lambda_comp3 = torch.where(sum_lambda_comp3<0.01, torch.tensor([1.]).to(torch.device('cuda')),sum_lambda_comp3) #将所有0替换为1 防止/0=nan
        return avg_lambda_comp3/sum_lambda_comp3*comp3_score
