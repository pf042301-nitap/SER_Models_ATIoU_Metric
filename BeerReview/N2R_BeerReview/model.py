import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
Tensor = torch.Tensor

class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]


class Embedding(nn.Module):

    def __init__(self, vocab_size, embedding_dim, pretrained_embedding=None):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embedding is not None:
            self.embedding.weight.data = torch.from_numpy(pretrained_embedding)
        self.embedding.weight.requires_grad = False

    def forward(self, x):
        """
        Inputs:
        x -- (batch_size, seq_length)
        Outputs
        shape -- (batch_size, seq_length, embedding_dim)
        """
        return self.embedding(x)


class Perturb_model(nn.Module):
    def __init__(self, args):
        super(Perturb_model, self).__init__()
        self.lay=True
        self.args = args
        self.embedding_layer = Embedding(args.vocab_size,
                                         args.embedding_dim,
                                         args.pretrained_embedding)

        self.gen = nn.GRU(input_size=args.embedding_dim,
                                  hidden_size=args.hidden_dim // 2,
                                  num_layers=args.num_layers,
                                  batch_first=True,
                                  bidirectional=True)
        if args.fr==1:
            self.cls=self.gen
        else:
            self.cls = nn.GRU(input_size=args.embedding_dim,
                              hidden_size=args.hidden_dim // 2,
                              num_layers=args.num_layers,
                              batch_first=True,
                              bidirectional=True)
        self.cls_fc = nn.Linear(args.hidden_dim, args.num_class)

        self.z_dim = 2
        self.gen_fc = nn.Linear(args.hidden_dim, self.z_dim)
        self.dropout = nn.Dropout(args.dropout)
        self.layernorm1 = nn.LayerNorm(args.hidden_dim)
        self.layernorm2 = nn.LayerNorm(args.hidden_dim)
        self.generator=nn.Sequential(self.gen,
                                     SelectItem(0),
                                     self.layernorm1,
                                     self.dropout,
                                     self.gen_fc)


    def perturb_gumbel(self,mask, logits: Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1, threshold_perturb: float =0.1) -> Tensor:
        """
        假设扰动比例为0.1
        总体思路：先按正常的generator得到被选中和没被选中的token，存在index里面.然后，在被选中的token里面取0.1比例标记，存在perturb_selected里面，比如有三个单词标记为需要翻转，则perturb_selected=【0，0，0，0，1，1，1，0，0.……】。
        计算在一个batch里面平均每个句子翻转了几个被选中的单词，然后在每个句子没被选中的单词中挑选出同样数量的单词标记为需要翻转，存在perturb_unselected里面
        perturb_all将所有需要翻转的位置置为1，否则为0
        """

        gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        )  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim)


        if hard:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]

            #perturb: 以一定概率翻转rationale的采样结果
            perturbation_sample = torch.FloatTensor(index.size()).to(index.device)
            perturbation_sample = torch.rand(index.size(),out=perturbation_sample)      #采样决定是否扰动

            # perturbation_sample = torch.rand(index.size())

            perturbation_threshold=torch.ones_like(perturbation_sample)*threshold_perturb     #扰动阈值
            perturbation_p=torch.cat((perturbation_threshold,perturbation_sample),dim=-1)
            perturbation_or_not=perturbation_p.min(dim,keepdim=True)[1].to(index.device)      #对整个句子采样，采样小于阈值即需要扰动（这里还没真正扰动，只是标记了），（即perturbation_or_not=1）

            perturb_selected=perturbation_or_not*index*mask         #被选中为rationale且未被mask的部分的token,再乘以扰动值
            num_of_perturb=int(torch.sum(perturb_selected)/perturb_selected.size()[0])    #选中的部分平均翻转了几个单词

            index_unselected=torch.ones_like(index)-index           #未被选中的部分为1
            index_unselected=index_unselected*mask                  #未被选中且未被mask的部分为1
            index_unselected=index_unselected.squeeze(-1).float()           #[batch,seq_len]
            num_unselected=(torch.sum(index_unselected,dim=1))
            if min(num_unselected)==0:
                perturb_all =perturb_selected
            else:
                try:
                    # index_perturb_unselected=index_unselected.multinomial(max(num_of_perturb,1))

                    index_perturb_unselected=torch.LongTensor([index_unselected.shape[0],max(num_of_perturb,1)]).to(index.device)
                    torch.multinomial(index_unselected,max(num_of_perturb,1),out=index_perturb_unselected)

                    perturb_unselected = torch.zeros_like(index_unselected).scatter_(1, index_perturb_unselected,
                                                                                     1).unsqueeze(
                        -1).long()  # 未被选中且未被mask的且被扰动的部分为1

                    perturb_all = torch.max(perturb_unselected, perturb_selected)  # 所有被扰动的部分
                except:
                    print('fail to sample unselected parts')
                    print(torch.prod(torch.sum(index_unselected,dim=1)))
                    print(torch.sum(index_unselected,dim=1))


            index2=abs(index-perturb_all) #perturb_all记录了所有需要扰动的位置，index记录了所有单词被选为rationale的情况。相减取绝对值代表，未被扰动的部分还是原来的值，扰动的部分就是0变1，1变0


            y_hard_perturb = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index2, 1.0)
            ret_perturb= y_hard_perturb - y_soft.detach() + y_soft      #用来计算预测分类准确率

            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft         #用来计算稀疏度和连续性
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret,ret_perturb

    def _independent_soft_sampling(self, rationale_logits):
        """
        Use the hidden states at all time to sample whether each word is a rationale or not.
        No dependency between actions. Return the sampled (soft) rationale mask.
        Outputs:
                z -- (batch_size, sequence_length, 2)
        """
        z = torch.softmax(rationale_logits, dim=-1)

        return z

    def independent_straight_through_sampling(self, rationale_logits):
        """
        Straight through sampling.
        Outputs:
            z -- shape (batch_size, sequence_length, 2)
        """
        # z = self._independent_soft_sampling(rationale_logits)
        z = F.gumbel_softmax(rationale_logits, tau=1, hard=True)
        return z

    def forward(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)    #[batch,seq_len,1]

        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        # gen_output, _ = self.gen(embedding)  # (batch_size, seq_length, hidden_dim)
        # if self.lay:
        #     gen_output = self.layernorm1(gen_output)
        # gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
        gen_logits=self.generator(embedding)
        ########## Sample ##########
        z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)
        ########## Classifier ##########
        cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))  # (batch_size, seq_length, embedding_dim)
        cls_outputs, _ = self.cls(cls_embedding)  # (batch_size, seq_length, hidden_dim)
        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))

        return z, cls_logits

    def forward_mochang(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)    #[batch,seq_len,1]

        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        # gen_output, _ = self.gen(embedding)  # (batch_size, seq_length, hidden_dim)
        # if self.lay:
        #     gen_output = self.layernorm1(gen_output)
        # gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
        gen_logits=self.generator(embedding)
        ########## Sample ##########
        z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)
        ########## Classifier ##########
        cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))  # (batch_size, seq_length, embedding_dim)
        cls_outputs, _ = self.cls(cls_embedding)  # (batch_size, seq_length, hidden_dim)
        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))

        return z, cls_logits,cls_outputs


    def train_one_step(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)
        # (batch_size, seq_length, embedding_dim)
        embedding = masks_ * self.embedding_layer(inputs)
        outputs, _ = self.cls(embedding)  # (batch_size, seq_length, hidden_dim)
        outputs = outputs * masks_ + (1. -
                                      masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        outputs = torch.transpose(outputs, 1, 2)
        outputs, _ = torch.max(outputs, axis=2)
        # shape -- (batch_size, num_classes)
        logits = self.cls_fc(self.dropout(outputs))
        return logits


    def g_skew(self,inputs, masks):
        #  masks_ (batch_size, seq_length, 1)
        masks_ = masks.unsqueeze(-1)
        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        gen_output, _ = self.gen(embedding)  # (batch_size, seq_length, hidden_dim)
        gen_output = self.layernorm1(gen_output)
        gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
        soft_log=self._independent_soft_sampling(gen_logits)
        return soft_log

    def perturb_forward(self, inputs, masks, perturb_rate):
        masks_ = masks.unsqueeze(-1)

        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        # gen_output, _ = self.gen(embedding)  # (batch_size, seq_length, hidden_dim)
        # if self.lay:
        #     gen_output = self.layernorm1(gen_output)
        # gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
        gen_logits = self.generator(embedding)
        ########## Sample ##########
        # z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)
        if perturb_rate==0:
            z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)
            z_perturb=z
        else:
            z,z_perturb=self.perturb_gumbel(masks_,gen_logits,tau=1, hard=True,threshold_perturb=perturb_rate)



        ########## Classifier ##########
        cls_embedding = embedding * (z_perturb[:, :, 1].unsqueeze(-1))  # (batch_size, seq_length, embedding_dim)
        cls_outputs, _ = self.cls(cls_embedding)  # (batch_size, seq_length, hidden_dim)
        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))
        return z, cls_logits

    def get_perturb_rationale(self, inputs, masks, perturb_rate):
        masks_ = masks.unsqueeze(-1)

        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        # gen_output, _ = self.gen(embedding)  # (batch_size, seq_length, hidden_dim)
        # if self.lay:
        #     gen_output = self.layernorm1(gen_output)
        # gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
        # gen_logits = self.generator(embedding)

        gen_emb, _ = self.gen(embedding)
        gen_emb = self.layernorm1(gen_emb)

        gen_logits = self.gen_fc(self.dropout(gen_emb))

        if perturb_rate == 0:
            z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)
            z_perturb = z
        else:
            z, z_perturb = self.perturb_gumbel(masks_, gen_logits, tau=1, hard=True, threshold_perturb=perturb_rate)

        return z, z_perturb

    def get_rationale(self, inputs, masks, perturb_rate):
        masks_ = masks.unsqueeze(-1)

        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        # gen_output, _ = self.gen(embedding)  # (batch_size, seq_length, hidden_dim)
        # if self.lay:
        #     gen_output = self.layernorm1(gen_output)
        # gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
        # gen_logits = self.generator(embedding)

        gen_emb, _ = self.gen(embedding)
        gen_emb = self.layernorm1(gen_emb)

        gen_logits = self.gen_fc(self.dropout(gen_emb))


        z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)



        return z

    def pred_with_rationale(self, inputs, masks, z):
        masks_ = masks.unsqueeze(-1)

        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))  # (batch_size, seq_length, embedding_dim)
        cls_outputs, _ = self.cls(cls_embedding)  # (batch_size, seq_length, hidden_dim)
        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))
        return cls_logits

    def grad(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)

        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        # gen_output, _ = self.gen(embedding)  # (batch_size, seq_length, hidden_dim)
        # if self.lay:
        #     gen_output = self.layernorm1(gen_output)
        # gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
        gen_logits=self.generator(embedding)
        ########## Sample ##########
        z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)
        ########## Classifier ##########
        embedding2=embedding.clone().detach()
        embedding2.requires_grad=True
        cls_embedding =embedding2  * (z[:, :, 1].unsqueeze(-1))  # (batch_size, seq_length, embedding_dim)
        cls_outputs, _ = self.cls(cls_embedding)  # (batch_size, seq_length, hidden_dim)
        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))
        return z, cls_logits,embedding2,cls_embedding


class Mochangmodel(nn.Module):
    def __init__(self, args):
        super(Mochangmodel, self).__init__()
        self.lay=True
        self.args = args
        self.embedding_layer = Embedding(args.vocab_size,
                                         args.embedding_dim,
                                         args.pretrained_embedding)

        self.gen = nn.GRU(input_size=args.embedding_dim,
                                  hidden_size=args.hidden_dim // 2,
                                  num_layers=args.num_layers,
                                  batch_first=True,
                                  bidirectional=True)
        if args.fr==1:
            self.cls=self.gen
        else:
            self.cls = nn.GRU(input_size=args.embedding_dim,
                              hidden_size=args.hidden_dim // 2,
                              num_layers=args.num_layers,
                              batch_first=True,
                              bidirectional=True)
        self.cls_fc = nn.Linear(args.hidden_dim, args.num_class)

        self.z_dim = 2
        self.gen_fc = nn.Linear(args.hidden_dim, self.z_dim)
        self.dropout = nn.Dropout(args.dropout)
        self.layernorm1 = nn.LayerNorm(args.hidden_dim)
        self.layernorm2 = nn.LayerNorm(args.hidden_dim)
        self.generator=nn.Sequential(self.gen,
                                     SelectItem(0),
                                     self.layernorm1,
                                     self.dropout,
                                     self.gen_fc)

        self.generator_with_presentation=nn.Sequential(
                                     self.dropout,
                                     self.gen_fc)


    def perturb_gumbel(self,mask, logits: Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1, threshold_perturb: float =0.1) -> Tensor:
        """
        假设扰动比例为0.1
        总体思路：先按正常的generator得到被选中和没被选中的token，存在index里面.然后，在被选中的token里面取0.1比例标记，存在perturb_selected里面，比如有三个单词标记为需要翻转，则perturb_selected=【0，0，0，0，1，1，1，0，0.……】。
        计算在一个batch里面平均每个句子翻转了几个被选中的单词，然后在每个句子没被选中的单词中挑选出同样数量的单词标记为需要翻转，存在perturb_unselected里面
        perturb_all将所有需要翻转的位置置为1，否则为0
        """


        gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        )  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim)


        if hard:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]

            #perturb: 以一定概率翻转rationale的采样结果
            perturbation_sample = torch.FloatTensor(index.size()).to(index.device)
            perturbation_sample = torch.rand(index.size(),out=perturbation_sample)      #采样决定是否扰动

            # perturbation_sample = torch.rand(index.size())

            perturbation_threshold=torch.ones_like(perturbation_sample)*threshold_perturb     #扰动阈值
            perturbation_p=torch.cat((perturbation_threshold,perturbation_sample),dim=-1)
            perturbation_or_not=perturbation_p.min(dim,keepdim=True)[1].to(index.device)      #对整个句子采样，采样小于阈值即需要扰动（这里还没真正扰动，只是标记了），（即perturbation_or_not=1）

            perturb_selected=perturbation_or_not*index*mask         #被选中为rationale且未被mask的部分的token,再乘以扰动值
            num_of_perturb=int(torch.sum(perturb_selected)/perturb_selected.size()[0])    #选中的部分平均翻转了几个单词

            index_unselected=torch.ones_like(index)-index           #未被选中的部分为1
            index_unselected=index_unselected*mask                  #未被选中且未被mask的部分为1
            index_unselected=index_unselected.squeeze(-1).float()           #[batch,seq_len]
            num_unselected=(torch.sum(index_unselected,dim=1))
            if min(num_unselected)==0:
                perturb_all =perturb_selected
            else:
                try:
                    # index_perturb_unselected=index_unselected.multinomial(max(num_of_perturb,1))

                    index_perturb_unselected=torch.LongTensor([index_unselected.shape[0],max(num_of_perturb,1)]).to(index.device)
                    torch.multinomial(index_unselected,max(num_of_perturb,1),out=index_perturb_unselected)

                    perturb_unselected = torch.zeros_like(index_unselected).scatter_(1, index_perturb_unselected,
                                                                                     1).unsqueeze(
                        -1).long()  # 未被选中且未被mask的且被扰动的部分为1

                    perturb_all = torch.max(perturb_unselected, perturb_selected)  # 所有被扰动的部分
                except:
                    print('fail to sample unselected parts')
                    print(torch.prod(torch.sum(index_unselected,dim=1)))
                    print(torch.sum(index_unselected,dim=1))





            index2=abs(index-perturb_all) #perturb_all记录了所有需要扰动的位置，index记录了所有单词被选为rationale的情况。相减取绝对值代表，未被扰动的部分还是原来的值，扰动的部分就是0变1，1变0


            y_hard_perturb = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index2, 1.0)
            ret_perturb= y_hard_perturb - y_soft.detach() + y_soft      #用来计算预测分类准确率




            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft         #用来计算稀疏度和连续性
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret,ret_perturb



    def independent_straight_through_sampling(self, rationale_logits):
        """
        Straight through sampling.
        Outputs:
            z -- shape (batch_size, sequence_length, 2)
        """

        z = F.gumbel_softmax(rationale_logits, tau=1, hard=True)
        return z

    def get_representation(self, inputs, masks, z=None):
        """
        Inputs:
            inputs -- (batch_size, seq_length)
            masks -- (batch_size, seq_length)
        Outputs:
            logits -- (batch_size, num_class)
        """
        # expand dim for masks
        masks_ = masks.unsqueeze(-1)
        # (batch_siz, seq_length, embedding_dim)
        embeddings = masks_ * self.embedding_layer(inputs)
        # (batch_siz, seq_length, embedding_dim * 2)
        if z is not None:
            embeddings = embeddings * (z.unsqueeze(-1))
        outputs, _ = self.cls(embeddings)

        # mask before max pooling
        outputs = outputs * masks_ + (1. - masks_) * (-1e6)

        # (batch_size, hidden_dim * 2, seq_length)
        outputs = torch.transpose(outputs, 1, 2)
        outputs, _ = torch.max(outputs, axis=2)
        # shape -- (batch_size, num_classes)
        logits = self.cls_fc(self.dropout(outputs))
        return logits,outputs

    def forward(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)    #[batch,seq_len,1]

        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        # gen_output, _ = self.gen(embedding)  # (batch_size, seq_length, hidden_dim)
        # if self.lay:
        #     gen_output = self.layernorm1(gen_output)
        # gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
        gen_logits=self.generator(embedding)
        ########## Sample ##########
        z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)
        ########## Classifier ##########
        cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))  # (batch_size, seq_length, embedding_dim)
        cls_outputs, _ = self.cls(cls_embedding)  # (batch_size, seq_length, hidden_dim)
        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))
        return z, cls_logits

    def forward_predemb(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)    #[batch,seq_len,1]

        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        # gen_output, _ = self.gen(embedding)  # (batch_size, seq_length, hidden_dim)
        # if self.lay:
        #     gen_output = self.layernorm1(gen_output)
        # gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
        gen_logits=self.generator(embedding)
        ########## Sample ##########
        z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)
        ########## Classifier ##########
        cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))  # (batch_size, seq_length, embedding_dim)
        cls_outputs, hid_state = self.cls(cls_embedding)  # cls_output:(batch_size, seq_length, hidden_dim), hid_state: (number_layer*direction, batch, hid_dim)


        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))
        return z, cls_logits,cls_outputs

    def mochang_forward(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)    #[batch,seq_len,1]

        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        # gen_output, _ = self.gen(embedding)  # (batch_size, seq_length, hidden_dim)
        # if self.lay:
        #     gen_output = self.layernorm1(gen_output)
        # gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
        gen_out,_=self.gen(embedding)
        gen_out=self.layernorm1(gen_out)

        gen_logits=self.generator_with_presentation(gen_out)
        ########## Sample ##########
        z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)


        gen_out_mochang=torch.detach(gen_out)
        gen_out_rationale=gen_out_mochang*(z[:, :, 1].unsqueeze(-1))* masks_
        sum_gen_out_rationale=gen_out_rationale.sum(dim=1)
        sum_mask=(z[:, :, 1]* masks).sum(dim=1).unsqueeze(-1)
        avg_gen_out_rationale=sum_gen_out_rationale/(sum_mask+1e-6)
        mochang=torch.norm(avg_gen_out_rationale, p=2, dim=1)


        ########## Classifier ##########
        cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))  # (batch_size, seq_length, embedding_dim)
        cls_outputs, _ = self.cls(cls_embedding)  # (batch_size, seq_length, hidden_dim)
        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))
        return z, cls_logits, mochang

    def train_one_step(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)
        # (batch_size, seq_length, embedding_dim)
        embedding = masks_ * self.embedding_layer(inputs)
        outputs, _ = self.cls(embedding)  # (batch_size, seq_length, hidden_dim)
        outputs = outputs * masks_ + (1. -
                                      masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        outputs = torch.transpose(outputs, 1, 2)
        outputs, _ = torch.max(outputs, axis=2)
        # shape -- (batch_size, num_classes)
        logits = self.cls_fc(self.dropout(outputs))
        return logits



    def g_skew(self,inputs, masks):
        #  masks_ (batch_size, seq_length, 1)
        masks_ = masks.unsqueeze(-1)
        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        gen_output, _ = self.gen(embedding)  # (batch_size, seq_length, hidden_dim)
        gen_output = self.layernorm1(gen_output)
        gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
        soft_log=self._independent_soft_sampling(gen_logits)
        return soft_log

    def perturb_forward(self, inputs, masks, perturb_rate):
        masks_ = masks.unsqueeze(-1)

        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        # gen_output, _ = self.gen(embedding)  # (batch_size, seq_length, hidden_dim)
        # if self.lay:
        #     gen_output = self.layernorm1(gen_output)
        # gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
        gen_logits = self.generator(embedding)
        ########## Sample ##########
        # z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)
        if perturb_rate==0:
            z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)
            z_perturb=z
        else:
            z,z_perturb=self.perturb_gumbel(masks_,gen_logits,tau=1, hard=True,threshold_perturb=perturb_rate)



        ########## Classifier ##########
        cls_embedding = embedding * (z_perturb[:, :, 1].unsqueeze(-1))  # (batch_size, seq_length, embedding_dim)
        cls_outputs, _ = self.cls(cls_embedding)  # (batch_size, seq_length, hidden_dim)
        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))
        return z, cls_logits

    def get_perturb_rationale(self, inputs, masks, perturb_rate):
        masks_ = masks.unsqueeze(-1)

        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        # gen_output, _ = self.gen(embedding)  # (batch_size, seq_length, hidden_dim)
        # if self.lay:
        #     gen_output = self.layernorm1(gen_output)
        # gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
        # gen_logits = self.generator(embedding)

        gen_emb, _ = self.gen(embedding)
        gen_emb = self.layernorm1(gen_emb)

        gen_logits = self.gen_fc(self.dropout(gen_emb))

        if perturb_rate == 0:
            z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)
            z_perturb = z
        else:
            z, z_perturb = self.perturb_gumbel(masks_, gen_logits, tau=1, hard=True, threshold_perturb=perturb_rate)

        return z, z_perturb

    def get_rationale(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)

        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        # gen_output, _ = self.gen(embedding)  # (batch_size, seq_length, hidden_dim)
        # if self.lay:
        #     gen_output = self.layernorm1(gen_output)
        # gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
        # gen_logits = self.generator(embedding)

        gen_emb, _ = self.gen(embedding)
        gen_emb = self.layernorm1(gen_emb)

        gen_logits = self.gen_fc(self.dropout(gen_emb))


        z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)



        return z

    def pred_with_rationale(self, inputs, masks, z):
        masks_ = masks.unsqueeze(-1)

        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))  # (batch_size, seq_length, embedding_dim)
        cls_outputs, _ = self.cls(cls_embedding)  # (batch_size, seq_length, hidden_dim)
        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))
        return cls_logits

    def grad(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)

        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        # gen_output, _ = self.gen(embedding)  # (batch_size, seq_length, hidden_dim)
        # if self.lay:
        #     gen_output = self.layernorm1(gen_output)
        # gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
        gen_logits=self.generator(embedding)
        ########## Sample ##########
        z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)
        ########## Classifier ##########
        embedding2=embedding.clone().detach()
        embedding2.requires_grad=True
        cls_embedding =embedding2  * (z[:, :, 1].unsqueeze(-1))  # (batch_size, seq_length, embedding_dim)
        cls_outputs, _ = self.cls(cls_embedding)  # (batch_size, seq_length, hidden_dim)
        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))
        return z, cls_logits,embedding2,cls_embedding

class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.args = args

        # initialize embedding layers
        self.embedding_layer = Embedding(args.vocab_size,
                                         args.embedding_dim,
                                         args.pretrained_embedding)
        # initialize a RNN encoder
        self.encoder = nn.GRU(input_size=args.embedding_dim,
                                  hidden_size=args.hidden_dim // 2,
                                  num_layers=args.num_layers,
                                  batch_first=True,
                                  bidirectional=True)
        # initialize a fc layer
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(args.hidden_dim, args.num_class)

    def forward(self, inputs, masks, z=None):
        """
        Inputs:
            inputs -- (batch_size, seq_length)
            masks -- (batch_size, seq_length)
        Outputs:
            logits -- (batch_size, num_class)
        """
        # expand dim for masks
        masks_ = masks.unsqueeze(-1)
        # (batch_siz, seq_length, embedding_dim)
        embeddings = masks_ * self.embedding_layer(inputs)
        # (batch_siz, seq_length, embedding_dim * 2)
        if z is not None:
            embeddings = embeddings * (z.unsqueeze(-1))
        outputs, _ = self.encoder(embeddings)

        # mask before max pooling
        outputs = outputs * masks_ + (1. - masks_) * (-1e6)

        # (batch_size, hidden_dim * 2, seq_length)
        outputs = torch.transpose(outputs, 1, 2)
        outputs, _ = torch.max(outputs, axis=2)
        # shape -- (batch_size, num_classes)
        logits = self.fc(self.dropout(outputs))
        return logits

    def get_representation(self, inputs, masks, z=None):
        """
        Inputs:
            inputs -- (batch_size, seq_length)
            masks -- (batch_size, seq_length)
        Outputs:
            logits -- (batch_size, num_class)
        """
        # expand dim for masks
        masks_ = masks.unsqueeze(-1)
        # (batch_siz, seq_length, embedding_dim)
        embeddings = masks_ * self.embedding_layer(inputs)
        # (batch_siz, seq_length, embedding_dim * 2)
        if z is not None:
            embeddings = embeddings * (z.unsqueeze(-1))
        outputs, _ = self.encoder(embeddings)

        # mask before max pooling
        outputs = outputs * masks_ + (1. - masks_) * (-1e6)

        # (batch_size, hidden_dim * 2, seq_length)
        outputs = torch.transpose(outputs, 1, 2)
        outputs, _ = torch.max(outputs, axis=2)
        # shape -- (batch_size, num_classes)
        logits = self.fc(self.dropout(outputs))
        return logits,outputs