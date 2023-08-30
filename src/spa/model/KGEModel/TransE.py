import torch.nn as nn
import torch
from .model import Model
from IPython import embed
from .relation_pattern import *


class TransE(Model):
    """`Translating Embeddings for Modeling Multi-relational Data`_ (TransE), which represents the relationships as translations in the embedding space.

    Attributes:
        args: Model configuration parameters.
        epsilon: Calculate embedding_range.
        margin: Calculate embedding_range and loss.
        embedding_range: Uniform distribution range.
        ent_emb: Entity embedding, shape:[num_ent, emb_dim].
        rel_emb: Relation embedding, shape:[num_rel, emb_dim].
    
    .. _Translating Embeddings for Modeling Multi-relational Data: http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-rela
    """
    def __init__(self, args):
        super(TransE, self).__init__(args)
        self.args = args
        self.ent_emb = None
        self.rel_emb = None
        # print(self.args.cuda_visible_devices)
        self.init_emb()

    def init_emb(self):
        """Initialize the entity and relation embeddings in the form of a uniform distribution.

        """
        self.epsilon = 2.0
        self.margin = nn.Parameter(
            torch.Tensor([self.args.margin]), 
            requires_grad=False
        )
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.margin.item() + self.epsilon) / self.args.emb_dim]), 
            requires_grad=False
        )

        self.ent_emb = nn.Embedding(self.args.num_ent, self.args.emb_dim)
        self.rel_emb = nn.Embedding(self.args.num_rel, self.args.emb_dim)
        nn.init.uniform_(tensor=self.ent_emb.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())
        nn.init.uniform_(tensor=self.rel_emb.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())
        

    def score_func(self, head_emb, relation_emb, tail_emb, mode):
        """Calculating the score of triples.
        
        The formula for calculating the score is :math:`\gamma - ||h + r - t||_F`

        Args:
            head_emb: The head entity embedding.
            relation_emb: The relation embedding.
            tail_emb: The tail entity embedding.
            mode: Choose head-predict or tail-predict.

        Returns:
            score: The score of triples.
        """
        score = (head_emb + relation_emb) - tail_emb
        score = self.margin.item() - torch.norm(score, p=1, dim=-1)
        return score

    def score_func_comp2(self, head_emb, relation_emb1, relation_emb2, tail_emb, mode):
        """Calculating the score of triples.
        
        The formula for calculating the score is :math:`\gamma - ||h + r - t||_F`

        Args:
            head_emb: The head entity embedding.
            relation_emb: The relation embedding.
            tail_emb: The tail entity embedding.
            mode: Choose head-predict or tail-predict.

        Returns:
            score: The score of triples.
        """
        score = (head_emb + relation_emb1 + relation_emb2) - tail_emb
        score = self.margin.item() - torch.norm(score, p=1, dim=-1)
        return score

    def score_func_comp3(self, head_emb, relation_emb1, relation_emb2, relation_emb3, tail_emb, mode):
        """Calculating the score of triples.
        
        The formula for calculating the score is :math:`\gamma - ||h + r - t||_F`

        Args:
            head_emb: The head entity embedding.
            relation_emb: The relation embedding.
            tail_emb: The tail entity embedding.
            mode: Choose head-predict or tail-predict.

        Returns:
            score: The score of triples.
        """
        score = (head_emb + relation_emb1 + relation_emb2 + relation_emb3) - tail_emb
        score = self.margin.item() - torch.norm(score, p=1, dim=-1)
        return score

    def forward(self, triples, negs=None, mode='single'):
        """The functions used in the training phase

        Args:
            triples: The triples ids, as (h, r, t), shape:[batch_size, 3].
            negs: Negative samples, defaults to None.
            mode: Choose head-predict or tail-predict, Defaults to 'single'.

        Returns:
            score: The score of triples.
        """
        head_emb, relation_emb, tail_emb = self.tri2emb(triples, negs, mode)
        score = self.score_func(head_emb, relation_emb, tail_emb, mode)

        return score

    def inv_relation(self, relation_emb):
        # import pdb;pdb.set_trace()
        return -relation_emb


    # def get_rel_inverse(self, relation_emb1, relation_emb2, comp2_rel_inv):
    #     relation_emb = torch.cat([relation_emb1, relation_emb2], 1)
    #     import pdb;pdb.set_trace()
    #     for i in range(len(comp2_rel_inv)):
    #         for j in range(2):
    #             if comp2_rel_inv[i][j]:
    #                 relation_emb[i][j] = self.inv_relation(relation_emb[i][j])

    #     return torch.chunk(relation_emb, 2, dim = 1)
    
    def get_score(self, batch, mode):
        """The functions used in the testing phase

        Args:
            batch: A batch of data.
            mode: Choose head-predict or tail-predict.

        Returns:
            score: The score of triples.
        """
        triples = batch["positive_sample"]
        head_emb, relation_emb, tail_emb = self.tri2emb(triples, mode=mode)
        score = self.score_func(head_emb, relation_emb, tail_emb, mode)
        
        if self.args.use_sym_weight:
            sym_score, JC_sym = relation_pattern.func_sym(self, batch, mode)
            # import pdb; pdb.set_trace()
            sym_score = self.args.lambda_sym * JC_sym * (sym_score - score)
            # print(sym_score)

        if self.args.use_inv_weight:
            inv_score, JC_inv = relation_pattern.func_inv(self, batch, mode)
            inv_score = self.args.lambda_inv * JC_inv * (inv_score - score)

        if self.args.use_sub_weight:
            sub_score, JC_sub = relation_pattern.func_sub(self, batch, mode)
            sub_score = self.args.lambda_sub * JC_sub * (sub_score - score)

        if self.args.use_comp2_weight:
            comp2_score, JC_comp2 = relation_pattern.func_comp2(self, batch, mode)
            # import pdb; pdb.set_trace()
            comp2_score = self.args.lambda_comp2 * JC_comp2 * (comp2_score - score)

        if self.args.use_comp3_weight:
            comp3_score, JC_comp3 = relation_pattern.func_comp3(self, batch, mode)
            comp3_score = self.args.lambda_comp3 * JC_comp3 * (comp3_score - score )

        # import pdb; pdb.set_trace()
        if self.args.use_sym_weight: score += sym_score
        if self.args.use_inv_weight: score += inv_score
        if self.args.use_sub_weight: score += sub_score
        if self.args.use_comp2_weight: score += comp2_score
        if self.args.use_comp3_weight: score += comp3_score
        
        return score
