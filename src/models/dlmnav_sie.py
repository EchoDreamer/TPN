import torch
import torch.nn as nn
from src.models.base_model import BaseEncoder
import numpy as np


class Encoder(BaseEncoder):
    def __init__(self, config, model, cls_token_id=0, sep_token_id=0, markers=True,device='cuda',combination='mean',gamma_pos=1,nota_length=10,
                 dropout=0.1,support_proto_counts=10,
                 nota_transform_multi=False):
        super().__init__(config=config, model=model, exemplar_method=self.sie_mnav, cls_token_id=cls_token_id, sep_token_id=sep_token_id,
                         markers=markers,device=device,combination=combination,gamma_pos=gamma_pos,nota_length=nota_length,
                         dropout=dropout,
                         support_proto_counts=support_proto_counts,
                         nota_transform_multi=nota_transform_multi)
    def sie_mnav(self,
                input_ids=None,
                attention_mask=None,
                entity_positions=None,
                labels=None, 
                type_labels=None,
                inputs_embeds_support=None,
                inputs_embeds_query=None):
        num_exemplars = input_ids.size(-2)
        batch_size = input_ids.size(0)
        if inputs_embeds_support is None:
            sequence_output, attention = self.encode(input_ids.view(-1, input_ids.size(-1)), attention_mask.view(-1, attention_mask.size(-1)))
        else:
            sequence_output, attention = self.encode(inputs_embeds_support,attention_mask.view(-1, attention_mask.size(-1)),adv=True)
        sequence_output = sequence_output.view(-1, num_exemplars, sequence_output.size(-2), sequence_output.size(-1))
        attention=attention.view(-1, num_exemplars, 12,attention.size(-2), attention.size(-1))
        batch_exemplars = []
        batch_exemplars2=[]
        batch_label_ids = []
        batch_label_types = []
        for batch_i in range(batch_size):
            episode_label_ids = []
            episode_label_types = []
            entity_embeddings = [[] for _ in entity_positions[batch_i]]
            attention_embeddings=[[] for _ in entity_positions[batch_i]]
            relation_embeddings = []
            relation_embeddings2=[]
            label_ids, label_types = [], []
            for batch_item in labels[batch_i]:
                li_in_batch = []
                lt_in_batch = []
                for l_h, l_t, l_r in batch_item:
                    li_in_batch.append((l_h, l_t))
                    lt_in_batch.append(l_r)
                label_ids.append(li_in_batch)
                label_types.append(lt_in_batch)
            rts = []

            for i, batch_item in enumerate(entity_positions[batch_i]):
                for entity in batch_item:
                    mention_embeddings = []
                    mention_attention=[]
                    for mention in entity:
                        if self.markers:
                            m_e = sequence_output[batch_i,i,mention[0],:]
                            a_e=attention[batch_i,i,:,mention[0],:]
                        else:
                            m_e = torch.mean(sequence_output[batch_i,i,mention[0]:mention[1],:], 0)
                            a_e=torch.mean(attention[batch_i,i,:,mention[0]:mention[1],:], 0)
                        mention_embeddings.append(m_e)
                        mention_attention.append(a_e)
                    if self.combination=='mean':
                        e_e = torch.mean(torch.stack(mention_embeddings, 0), 0)
                        a_a=torch.mean(torch.stack(mention_attention, 0), 0)
                    else :
                        e_e=torch.logsumexp(torch.stack(mention_embeddings, 0), 0)
                        a_a=torch.mean(torch.stack(mention_attention, 0), 0)
                    entity_embeddings[i].append(e_e)
                    attention_embeddings[i].append(a_a)
                for i_h, h in enumerate(entity_embeddings[i]):
                    for i_t, t in enumerate(entity_embeddings[i]):
                        if i_h == i_t:
                            continue
                        if (i_h, i_t) in label_ids[i]:
                            episode_label_ids.append(len(relation_embeddings))
                            types_for_label = []
                            for li, lt in zip(label_ids[i], label_types[i]):
                                if li == (i_h, i_t):
                                    types_for_label.append(lt)
                                    rts.append(lt)
                            episode_label_types.append(types_for_label)
                        else:
                            episode_label_ids.append(len(relation_embeddings))
                            episode_label_types.append(["NOTA"])
                        atten_mul=attention_embeddings[i][i_h].mul(attention_embeddings[i][i_t])
                        atten_mul=atten_mul.mean(0)
                        H_atten_temp=atten_mul/(atten_mul.sum(0,keepdim=True)+1e-5)
                        atten_all_sentence=torch.matmul(H_atten_temp,sequence_output[batch_i,i,:,:]).reshape(-1)

                        relation_embeddings.append(torch.cat((h,atten_all_sentence),dim=0))
                        relation_embeddings2.append(torch.cat((t,atten_all_sentence),dim=0))
            batch_exemplars.append(torch.stack(relation_embeddings,dim=0))

            batch_exemplars2.append(torch.stack(relation_embeddings2,dim=0))
            batch_label_ids.append(episode_label_ids)
            batch_label_types.append(episode_label_types)
        

        max_len = max((l.size(0) for l in batch_exemplars))
        batch_exemplars = list(map(lambda l:torch.cat((l,torch.tensor([[0]*768*2]*(max_len - len(l)),dtype=torch.float32,device=self.device)),dim=0), batch_exemplars))
        batch_exemplars = torch.stack(batch_exemplars,dim=0)
        max_len = max((l.size(0) for l in batch_exemplars2))
        batch_exemplars2 = list(map(lambda l:torch.cat((l,torch.tensor([[0]*768*2]*(max_len - len(l)),dtype=torch.float32,device=self.device)),dim=0), batch_exemplars2))
        batch_exemplars2 = torch.stack(batch_exemplars2,dim=0)
        batch_exemplars=torch.cat([torch.tanh(self.dropout_layer(self.head_extractor(batch_exemplars))),torch.tanh(self.dropout_layer(self.tail_extractor(batch_exemplars2)))],dim=-1)

        batch_exemplars_float32=batch_exemplars.type(torch.float32)
        batch_prototypes = []
        for exemplars, label_ids, label_types, type_index in zip(batch_exemplars_float32, batch_label_ids, batch_label_types, type_labels):
            
            episodes_prototypes = [None for _ in type_index]

            for relation_type in type_index:
                embeddings = []
                for i, t in zip(label_ids, label_types):
                    if relation_type in t:
                        embeddings.append(exemplars[i])

                embeddings = torch.stack(embeddings, 0) 

                if relation_type != "NOTA":
                    index_emb=torch.randperm(embeddings.size(0))
                    embdding=embeddings[index_emb[0:self.support_proto_counts],:]
                    episodes_prototypes[type_index.index(relation_type)] = embdding
                else :
                    indexes = torch.randperm(embeddings.shape[0])
                    nota_=embeddings[indexes[:self.nota_length], :]
                    if self.nota_transform_multi:
                        episodes_prototypes[type_index.index("NOTA")]=self.nota_transform_combine(self.nota_transform_first,self.nota_transform_second,nota_)
                    else:
                        episodes_prototypes[type_index.index("NOTA")]=torch.tanh(self.nota_transform(nota_)) 
            batch_prototypes.append(episodes_prototypes)
        return batch_prototypes
    
    def nota_transform_combine(self,first_layer,second_layer,content):
        x=self.nota_dropout_layer(content)
        after_first_layer=torch.tanh(first_layer(x))
        return second_layer(after_first_layer)
    

