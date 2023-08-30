import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import Model
from .relation_pattern import *

class PairRE(Model):
    """`PairRE: Knowledge Graph Embeddings via Paired Relation Vectors`_ (PairRE), which paired vectors for each relation representation to model complex patterns.

    Attributes:
        args: Model configuration parameters.
        epsilon: Calculate embedding_range.
        margin: Calculate embedding_range and loss.
        embedding_range: Uniform distribution range.
        ent_emb: Entity embedding, shape:[num_ent, emb_dim].
        rel_emb: Relation embedding, shape:[num_rel, emb_dim * 2].
    
    .. _PairRE: Knowledge Graph Embeddings via Paired Relation Vectors: https://arxiv.org/pdf/2011.03798.pdf
    """
    def __init__(self, args):
        super(PairRE, self).__init__(args)
        self.args = args
        self.ent_emb = None
        self.rel_emb = None
        self.init_emb()

    def init_emb(self):
        """Initialize the entity and relation embeddings in the form of a uniform distribution.

        """
        self.epsilon = 2.0
        self.margin = nn.Parameter(
            torch.Tensor([self.args.margin]),
            requires_grad=False,
        )
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.margin.item() + self.epsilon) / self.args.emb_dim]),
            requires_grad=False,
        )
        self.ent_emb = nn.Embedding(self.args.num_ent, self.args.emb_dim)
        nn.init.uniform_(
            tensor=self.ent_emb.weight.data,
            a = -self.embedding_range.item(),
            b = self.embedding_range.item(),
        )
        self.rel_emb = nn.Embedding(self.args.num_rel, self.args.emb_dim * 2)
        nn.init.uniform_(
            tensor=self.rel_emb.weight.data,
            a = -self.embedding_range.item(),
            b = self.embedding_range.item(), 
        )

    def score_func(self, head_emb, relation_emb, tail_emb, mode):
        """Calculating the score of triples.
        
        The formula for calculating the score is :math:`\gamma - || h \circ r^H - t  \circ r^T ||`

        Args:
            head_emb: The head entity embedding.
            relation_emb: The relation embedding.
            tail_emb: The tail entity embedding.
            mode: Choose head-predict or tail-predict.

        Returns:
            score: The score of triples.
        """
        re_head, re_tail = torch.chunk(relation_emb, 2, dim=2)

        head = F.normalize(head_emb, 2, -1)
        tail = F.normalize(tail_emb, 2, -1)

        score = head * re_head - tail  * re_tail
        return self.margin.item() - torch.norm(score, p=1, dim=2)
    
    def inv_relation(self, relation_emb):
        # import pdb;pdb.set_trace()
        re_head, re_tail = torch.chunk(relation_emb, 2, dim=-1)
        inv_relation_emb = torch.cat([re_tail,re_head], dim=-1)

        return inv_relation_emb

    def score_func_comp2(self, head_emb, relation_emb1, relation_emb2, tail_emb, mode):
        re_head1, re_tail1 = torch.chunk(relation_emb1, 2, dim=2)
        re_head2, re_tail2 = torch.chunk(relation_emb2, 2, dim=2)

        head = F.normalize(head_emb, 2, -1)
        tail = F.normalize(tail_emb, 2, -1)

        score = head * re_head1 * re_head2 - tail  * re_tail1 * re_tail2
        return self.margin.item() - torch.norm(score, p=1, dim=2)
    
    def score_func_comp3(self, head_emb, relation_emb1, relation_emb2, relation_emb3, tail_emb, mode):
        re_head1, re_tail1 = torch.chunk(relation_emb1, 2, dim=2)
        re_head2, re_tail2 = torch.chunk(relation_emb2, 2, dim=2)
        re_head3, re_tail3 = torch.chunk(relation_emb3, 2, dim=2)

        head = F.normalize(head_emb, 2, -1)
        tail = F.normalize(tail_emb, 2, -1)

        score = head * re_head1 * re_head2 * re_head3 - tail  * re_tail1 * re_tail2 * re_tail3
        return self.margin.item() - torch.norm(score, p=1, dim=2)

    def forward(self, triples, negs=None, mode='single'):
        """The functions used in the training phase

        Args:
            triples: The triples ids, as (h, r, t), shape:[batch_size, 3].
            negs: Negative samples, defaults to None.
            mode: Choose head-predict or tail-predict, Defaults to 'single'.

        Returns:
            score: The score of triples.
        """
        head_emb, relation_emb, tail_emb = self.tri2emb(triples, negs, mode=mode)
        score = self.score_func(head_emb, relation_emb, tail_emb, mode)
        return score
    
    def get_score(self, batch, mode):
        """The functions used in the testing phase

        Args:
            batch: A batch of data.
            mode: Choose head-predict or tail-predict.

        Returns:
            score: The score of triples.
        """
        triples = batch['positive_sample']
        head_emb, relation_emb, tail_emb = self.tri2emb(triples, mode=mode)
        score = self.score_func(head_emb, relation_emb, tail_emb, mode)
        
        if self.args.use_sym_weight:
            sym_score, JC_sym = relation_pattern.func_sym(self, batch, mode)
            score = score + self.args.lambda_sym * JC_sym * (sym_score - score)

        if self.args.use_inv_weight:
            inv_score, JC_inv = relation_pattern.func_inv(self, batch, mode)
            score = score + self.args.lambda_inv * JC_inv * (inv_score - score)

        if self.args.use_sub_weight:
            sub_score, JC_sub = relation_pattern.func_sub(self, batch, mode)
            score = score + self.args.lambda_sub * JC_sub * (sub_score - score)

        if self.args.use_comp2_weight:
            comp2_score, JC_comp2 = relation_pattern.func_comp2(self, batch, mode)
            score = score + self.args.lambda_comp2 * JC_comp2 * ( comp2_score - score )

        if self.args.use_comp3_weight:
            comp3_score, JC_comp3 = relation_pattern.func_comp3(self, batch, mode)
            score = score + self.args.lambda_comp3 * JC_comp3 * ( comp3_score - score )

        return score