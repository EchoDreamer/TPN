from src.models.util import process_long_input
from random import sample
import torch
import torch.nn as nn
from transformers import BertConfig, RobertaConfig, DistilBertConfig, XLMRobertaConfig
import numpy as np
import torch.nn.functional as F
from src.models.losses import ATLoss_cal
class BaseEncoder(nn.Module):
    def __init__(self, config, model, exemplar_method, cls_token_id=0, sep_token_id=0, markers=True,device='cuda',combination='mean',gamma_pos=1
                 ,nota_length=10,dropout=0.1,
                 support_proto_counts=10,
                 nota_transform_multi=False):
        super().__init__()
        self.support_proto_counts=support_proto_counts
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size
        self.markers = markers
        self.gamma_pos= gamma_pos
        self.nota_length=nota_length
        self.nota_embeddings = nn.Parameter(torch.zeros((20,1536)))
        torch.nn.init.uniform_(self.nota_embeddings, a=-1.0, b=1.0)
        self.nota_rel_embeddings=nn.Parameter(torch.zeros((1,1536)))
        torch.nn.init.uniform_(self.nota_rel_embeddings, a=-1.0, b=1.0)
        
        self.first_run = True
        self.cross_domain = False
        self.device=device
        self.nota_transform=nn.Linear(in_features=2 * config.hidden_size, out_features=2*config.hidden_size,bias=False)
        
        
        self.nota_transform_first=nn.Linear(in_features=2 * config.hidden_size, out_features=4*config.hidden_size,bias=False)
        self.nota_transform_second=nn.Linear(in_features=4 * config.hidden_size, out_features=2*config.hidden_size,bias=False)
        
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.combination = combination
        self.set_exemplars = exemplar_method
        self.dropout=dropout
        self.dropout_layer=nn.Dropout(p=self.dropout)
        self.head_extractor = nn.Linear(in_features=2 * config.hidden_size, out_features=config.hidden_size,bias=False)
        self.tail_extractor = nn.Linear(in_features=2 * config.hidden_size, out_features=config.hidden_size,bias=False)
        self.relation_transform=nn.Linear(in_features=config.hidden_size, out_features=2*config.hidden_size,bias=False)
        self.max_label_instance=0
        self.nota_transform_multi=nota_transform_multi
        self.nota_dropout_layer=torch.nn.Dropout(p=0.1)

    def encode(self, input_ids, attention_mask,adv=False):
        config = self.config
        if type(config) == BertConfig or type(config) == DistilBertConfig:
            start_tokens = [self.cls_token_id]
            end_tokens = [self.sep_token_id]
        elif type(config) == RobertaConfig or type(config) == XLMRobertaConfig:
            start_tokens = [self.cls_token_id]
            end_tokens = [self.sep_token_id, self.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens,adv)
        return sequence_output, attention
    def forward(self,
                exemplar_input_ids=None,
                exemplar_masks=None,
                exemplar_entity_positions=None,
                exemplar_labels=None,
                query_input_ids=None,
                query_masks=None,
                query_entity_positions=None,
                query_labels=None,
                type_labels=None,
                inputs_embeds_support=None,
                inputs_embeds_query=None
                ):
        prototypes= self.set_exemplars(exemplar_input_ids, 
                                        exemplar_masks,
                                        exemplar_entity_positions, 
                                        exemplar_labels, 
                                        type_labels, 
                                        inputs_embeds_support=inputs_embeds_support,
                                        inputs_embeds_query=inputs_embeds_query)
        nota_id = [x.index("NOTA") for x in type_labels]
        num_queries = query_input_ids.size(-2)
        batch_size = query_input_ids.size(0)
        labels = None
        if query_labels is not None:
            labels = []
            for batch_i in range(batch_size):
                labels_in_episode = []
                for i, query in enumerate(query_labels[batch_i]):
                    num_entities_in_query = len(query_entity_positions[batch_i][i])
                    labels_in_query = torch.zeros((num_entities_in_query, num_entities_in_query, len(type_labels[batch_i])), device="cuda")
                    labels_in_query[:,:, nota_id[batch_i]] = 1
                    for k in range(num_entities_in_query):
                        labels_in_query[k,k, nota_id[batch_i]] = 0
                    for l_h, l_t, l_r in query:
                        labels_in_query[l_h,l_t, nota_id[batch_i]] = 0
                        labels_in_query[l_h,l_t, type_labels[batch_i].index(l_r)] = 1
                    labels_in_episode.append(labels_in_query)
                labels.append(labels_in_episode)

        if inputs_embeds_query==None:
            sequence_output, attention = self.encode(query_input_ids.view(-1, query_input_ids.size(-1)), query_masks.view(-1, query_masks.size(-1)))
        else:
            sequence_output, attention = self.encode(inputs_embeds_query, query_masks.view(-1, query_masks.size(-1)),adv=True)
        sequence_output = sequence_output.view(-1, num_queries, sequence_output.size(-2), sequence_output.size(-1))
        attention=attention.view(-1, num_queries, 12,attention.size(-2), attention.size(-1))
        all_matches = []
        loss = 0
        candidates = 0
        for batch_i in range(batch_size):

            entity_embeddings = [[] for _ in query_entity_positions[batch_i]]
            entity_attention = [[] for _ in query_entity_positions[batch_i]]
            matches = [[] for _ in query_entity_positions[batch_i]]

            for i, batch_item in enumerate(query_entity_positions[batch_i]):
                
                num_entities_in_query = len(batch_item)
                predictions_for_query = torch.zeros((num_entities_in_query, num_entities_in_query, len(type_labels[batch_i])), device="cuda")
                mask = torch.ones((num_entities_in_query, num_entities_in_query, len(type_labels[batch_i])), dtype=torch.bool, device="cuda")
                for k in range(num_entities_in_query):
                    mask[k,k, :] = 0

                for entity in batch_item:
                    mention_embeddings = []
                    mention_attention = []
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
                        a_a=torch.logsumexp(torch.stack(mention_attention, 0), 0)
                    # e_e = torch.mean(torch.stack(mention_embeddings, 0), 0)
                    entity_embeddings[i].append(e_e)
                    entity_attention[i].append(a_a)
                embs = torch.stack(entity_embeddings[i])
                attens=torch.stack(entity_attention[i])
                h_attentions=torch.unsqueeze(attens, 1)
                h_attentions = h_attentions.repeat(1, h_attentions.size()[0],1, 1)
                t_attentions=h_attentions.transpose(0,1)

                mul_attentions=(h_attentions*t_attentions).mean(dim=-2)
                mul_attentions=mul_attentions/mul_attentions.sum(dim=-1, keepdim=True)

                all_atten_emb=torch.matmul(mul_attentions,sequence_output[batch_i,i,:,:])
                all_atten_emb=all_atten_emb.flatten(0,1)
                h_entities = torch.unsqueeze(embs, 1)

                h_entities = h_entities.repeat(1, h_entities.size()[0],1)
                t_entities = h_entities.transpose(0,1)


                # flatten
                h_entities, t_entities = h_entities.flatten(0,1), t_entities.flatten(0,1)



                h_all=torch.tanh(self.dropout_layer(self.head_extractor(torch.cat((h_entities,all_atten_emb),dim=-1))))

                t_all=torch.tanh(self.dropout_layer(self.tail_extractor(torch.cat((t_entities,all_atten_emb),dim=-1))))
                candidates = torch.cat((h_all, t_all), dim=-1)
                scores = []
                

                for class_prototypes in prototypes[batch_i]:
                        
                    class_scores = candidates.unsqueeze(0) * class_prototypes.unsqueeze(1)
                    class_scores = torch.sum(class_scores, dim=-1)

                    class_scores_max = class_scores.max(dim=0,keepdim=False)[0]

                    scores.append(class_scores_max) 
                        
                        
                        
                        
                
                
                scores = torch.stack(scores).swapaxes(0,1)

                with torch.no_grad():
                    predictions_binary = ATLoss_cal(self.gamma_pos).get_label(scores.detach(), num_labels=1).view((num_entities_in_query, num_entities_in_query, len(type_labels[batch_i])))
                if self.training:
                    predictions_for_query = scores.view((num_entities_in_query, num_entities_in_query, len(type_labels[batch_i])))
                else:
                    predictions_for_query = scores.view((num_entities_in_query, num_entities_in_query, len(type_labels[batch_i])))
            
                with torch.no_grad():
                    for i_h, h in enumerate(entity_embeddings[i]):
                        for i_t, t in enumerate(entity_embeddings[i]):
                            if i_h == i_t:
                                continue

                            for rt in range(len(prototypes[batch_i])):
                                if rt == nota_id[batch_i]:
                                    continue
                             
                                if predictions_binary[i_h, i_t, rt] == 1.0:
                                    matches[i].append([i_h,i_t,type_labels[batch_i][rt]])

                # ------- LOSS CALCULATION --------
                if query_labels is not None:

                    loss += ATLoss_cal(self.gamma_pos)(torch.masked_select(predictions_for_query, mask).view(-1, predictions_for_query.size(-1)), torch.masked_select(labels[batch_i][i].float(), mask).view(-1, predictions_for_query.size(-1)))

            all_matches.append(matches)
        if query_labels is not None:
            if inputs_embeds_support is not None:
                loss = loss / batch_size
            return all_matches, loss
        return all_matches


