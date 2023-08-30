import torch.nn as nn
import torch
from .model import Model
from IPython import embed
from .relation_pattern import *

class RotatE(Model):
    """`RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space`_ (RotatE), which defines each relation as a rotation from the source entity to the target entity in the complex vector space.

    Attributes:
        args: Model configuration parameters.
        epsilon: Calculate embedding_range.
        margin: Calculate embedding_range and loss.
        embedding_range: Uniform distribution range.
        ent_emb: Entity embedding, shape:[num_ent, emb_dim * 2].
        rel_emb: Relation_embedding, shape:[num_rel, emb_dim].

    .. _RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space: https://openreview.net/forum?id=HkgEQnRqYQ
    """
    def __init__(self, args):
        super(RotatE, self).__init__(args)
        self.args = args
        self.ent_emb = None
        self.rel_emb = None

        self.init_emb()

    def init_emb(self):
        """Initialize the entity and relation embeddings in the form of a uniform distribution."""
        self.epsilon = 2.0
        self.margin = nn.Parameter(
            torch.Tensor([self.args.margin]), 
            requires_grad=False
        )
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.margin.item() + self.epsilon) / self.args.emb_dim]), 
            requires_grad=False
        )
        
        self.ent_emb = nn.Embedding(self.args.num_ent, self.args.emb_dim * 2)
        self.rel_emb = nn.Embedding(self.args.num_rel, self.args.emb_dim)
        nn.init.uniform_(tensor=self.ent_emb.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())
        nn.init.uniform_(tensor=self.rel_emb.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())
        

    def score_func(self, head_emb, relation_emb, tail_emb, mode):
        """Calculating the score of triples.

        The formula for calculating the score is :math:`\gamma - \|h \circ r - t\|`

        Args:
            head_emb: The head entity embedding.
            relation_emb: The relation embedding.
            tail_emb: The tail entity embedding.
            mode: Choose head-predict or tail-predict.

        Returns:
            score: The score of triples.
        """
        pi = 3.14159265358979323846
        re_head, im_head = torch.chunk(head_emb, 2, dim=-1)
        re_tail, im_tail = torch.chunk(tail_emb, 2, dim=-1)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation_emb/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)
        score = self.margin.item() - score.sum(dim = -1)
        return score

    def inv_relation(self, relation_emb):
        # import pdb;pdb.set_trace()
        return -relation_emb

    def score_func_comp2(self, head_emb, relation_emb1, relation_emb2, tail_emb, mode):
        pi = 3.14159265358979323846
        re_head, im_head = torch.chunk(head_emb, 2, dim=-1)
        re_tail, im_tail = torch.chunk(tail_emb, 2, dim=-1)
        
        phase_relation = (relation_emb1 + relation_emb2)/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)
        score = self.margin.item() - score.sum(dim = -1)
        return score
    
    def score_func_comp3(self, head_emb, relation_emb1, relation_emb2, relation_emb3, tail_emb, mode):
        pi = 3.14159265358979323846
        re_head, im_head = torch.chunk(head_emb, 2, dim=-1)
        re_tail, im_tail = torch.chunk(tail_emb, 2, dim=-1)
        
        phase_relation = (relation_emb1 + relation_emb2 + relation_emb3)/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)
        score = self.margin.item() - score.sum(dim = -1)
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