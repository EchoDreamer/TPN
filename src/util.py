import torch
class FreeLB(object):
    def __init__(self, adv_K, adv_lr, adv_init_mag, adv_max_norm=0., adv_norm_type='l2', base_model='bert'):
        self.adv_K = adv_K
        self.adv_lr = adv_lr    
        self.adv_max_norm = adv_max_norm   
        self.adv_init_mag = adv_init_mag   
        self.adv_norm_type = adv_norm_type  
        self.base_model = base_model        
    def attack(self, model, inputs, gradient_accumulation_steps=1,scaler=None):
        """  
                        
                inputs_freelb={
                    "exemplar_input_ids":exemplar_tokens.to(device_debug),
                    "exemplar_masks":exemplar_mask.to(device_debug),
                    "exemplar_entity_positions":exemplar_positions,
                    "exemplar_labels":exemplar_labels,
                    "query_input_ids":query_tokens.to(device_debug),
                    "query_masks":query_mask.to(device_debug),
                    "query_entity_positions":query_positions,
                    "query_labels":query_labels,
                    "type_labels":label_types,
                }
        """
        embeds_init_support = model.model.embeddings.word_embeddings(inputs["exemplar_input_ids"].view(-1, inputs["exemplar_input_ids"].size(-1)))
        embeds_init_query=model.model.embeddings.word_embeddings(inputs["query_input_ids"].view(-1, inputs["query_input_ids"].size(-1)))
        delta_support = torch.zeros_like(embeds_init_support)  
        delta_query=torch.zeros_like(embeds_init_query)
        loss,logits= None,None
        for astep in range(self.adv_K):
            delta_support.requires_grad_()
            delta_query.requires_grad_()
            inputs['inputs_embeds_support'] = delta_support + embeds_init_support  
            inputs['inputs_embeds_query']=delta_query+embeds_init_query
            logits,loss= model(**inputs)
            loss = loss / gradient_accumulation_steps
            scaler.scale(loss).backward()
            delta_grad_support = delta_support.grad.clone().detach() 
            delta_grad_query=delta_query.grad.clone().detach()
            if self.adv_norm_type == "l2":
                denorm_support = torch.norm(delta_grad_support.view(delta_grad_support.size(0), -1), dim=1).view(-1, 1, 1)
                denorm_support = torch.clamp(denorm_support, min=1e-8)
                delta_support = (delta_support + self.adv_lr * delta_grad_support / denorm_support).detach()
                denorm_query=torch.norm(delta_grad_query.view(delta_grad_query.size(0),-1),dim=1).view(-1,1,1)
                denorm_query=torch.clamp(denorm_query,min=1e-8)
                delta_query=(delta_query+self.adv_lr*delta_grad_query/denorm_query).detach()
                if self.adv_max_norm > 0:
                    delta_norm_support = torch.norm(delta_support.view(delta_support.size(0), -1).float(), p=2, dim=1).detach()
                    exceed_mask_support = (delta_norm_support > self.adv_max_norm).to(embeds_init_support)
                    reweights_support = (self.adv_max_norm / delta_norm_support * exceed_mask_support + (1 - exceed_mask_support)).view(-1, 1, 1)
                    delta_support = (delta_support * reweights_support).detach()
                    delta_norm_query=torch.norm(delta_query.view(delta_query.size(0),-1).float(),p=2,dim=1).detach()
                    exceed_mask_query=(delta_norm_query>self.adv_max_norm).to(embeds_init_query)
                    reweights_query=(self.adv_max_norm/delta_norm_query*exceed_mask_query+(1-exceed_mask_query)).view(-1,1,1)
                    delta_query=(delta_query*reweights_query).detach()
            embeds_init_support = model.model.embeddings.word_embeddings(inputs["exemplar_input_ids"].view(-1, inputs["exemplar_input_ids"].size(-1)))
            embeds_init_query=model.model.embeddings.word_embeddings(inputs["query_input_ids"].view(-1, inputs["query_input_ids"].size(-1)))
            
        return  logits,loss

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad) 
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
def get_f1(tp, fp, fn):


    if sum(tp.values())+sum(fp.values()) == 0:
        precision = 0
    else:
        precision = sum(tp.values())/(sum(tp.values())+sum(fp.values()))

    if sum(tp.values())+sum(fn.values()) == 0:
        recall = 0
    else:
        recall = sum(tp.values())/(sum(tp.values())+sum(fn.values()))

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return 100 * precision, 100 * recall, 100 * f1

def get_f1_macro(tp, fp, fn, prnt=False):
    if prnt:
        print('{:<10}  {:>10}  {:>10}  {:>10}    {:<10}'.format(f"type", f"precision", f"recall", f"f1", f"support"))

    p = []
    r = []
    f = []
    s = []
    for rtype in tp.keys():
        if tp[rtype]+fn[rtype] == 0:
            continue
        if tp[rtype]+fp[rtype] == 0:
            precision = 0
        else:
            precision = tp[rtype]/(tp[rtype]+fp[rtype])

        if tp[rtype]+fn[rtype] == 0:
            recall = 0
        else:
            recall = tp[rtype]/(tp[rtype]+fn[rtype])

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        support = tp[rtype]+fn[rtype]
        if prnt:
            print('{:<10}  {:>10}  {:>10}  {:>10}    {:<10}'.format(f"{rtype}", f"{100*precision:.2f}", f"{100*recall:.2f}", f"{100*f1:.2f}", f"{support}"))
        p.append(precision)
        r.append(recall)
        f.append(f1)
        s.append(support)

    if prnt:
        print('{:<10}  {:>10}  {:>10}  {:>10}    {:<10}'.format("-", "-", "-", "-", "-",))
        print('{:<10}  {:>10}  {:>10}  {:>10}'.format(f"macro", f"{100*sum(p)/len(p):.2f}", f"{100*sum(r)/len(r):.2f}", f"{100*sum(f)/len(f):.2f}"))
    
    return 100*sum(p)/len(p), 100*sum(r)/len(r), 100*sum(f)/len(f)
