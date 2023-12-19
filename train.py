# %%

from transformers import AutoTokenizer, AutoConfig, AutoModel
import string
from sklearn import metrics
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.data import parse_episodes, collate_fn_train, parse_episodes_from_index
import os
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
import torch
import torch.nn as nn
import wandb
import argparse
from src.models.dlmnav_sie import Encoder
import numpy as np
from src.models.util import set_seed
from src.util import get_f1, get_f1_macro,FGM,FreeLB
from tqdm import tqdm
import sys
import time 
import os
import pickle
os.environ["CUDA_VISIBLE_DEVICES"]="3"
if __name__ == "__main__":
    debug=True
    device_debug='cuda'
    random_string = ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(10))
    print(random_string) 
    time_begin=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    parser =argparse.ArgumentParser()
    parser.add_argument("--nota_transform_multi",type=bool,default=True,help="use nota_transform_multi loss")
    parser.add_argument("--freelb",type=bool,default=False,help="use freelb loss")
    parser.add_argument("--support_proto_counts",type=int,default=10,help="counts of support proto")
    parser.add_argument("--adv_K", type=int, default=3, help="number of steps for adversarial training")
    parser.add_argument("--adv_lr", type=float, default=1.5e-1, help="learning rate for adversarial training")
    parser.add_argument("--adv_max_norm", type=float, default=4.5e-1, help="max norm for adversarial training")
    parser.add_argument("--adv_init_mag", type=float, default=0, help="magnitude for initialization")
    parser.add_argument("--adv_norm_type", type=str, default="l2", help="norm type for adversarial training")
    parser.add_argument("--epsilon", type=float, default=0.2, help="epsilon for AT")
    parser.add_argument("--dropout", type=float, default=0., help="dropout")
    parser.add_argument("--weight_decay", type=float, default=0., help="weight decay")
    parser.add_argument("--seed_model", type=int, default=123, help="random seed for model")
    parser.add_argument("--nota_length", type=int, default=10, help="max length of nota")
    parser.add_argument("--gamma_pos", type=float, default=1, help="gamma for positive samples")
    parser.add_argument("--single_lr", type=bool, default=True, help="use single learning rate for all layers")
    parser.add_argument("--mention_combination", type=str, default="mean", help="The way mention is combined when calculating entity representation(logsumexp,Parameter,mean)")
    parser.add_argument("--use_markers", type=bool, default=True, help="use entity marker")
    parser.add_argument("--seed_data", type=int, default=123, help="random seed for data")
    parser.add_argument("--num_epochs", type=int, default=1, help="number of epochs to train")
    parser.add_argument("--support_docs_train", type=int, default=3, help="number of support documents during training")
    parser.add_argument("--support_docs_eval", type=int, default=3, help="number of support documents during eval")
    parser.add_argument("--query_docs_train", type=int, default=1, help="number of query documents (train)")
    parser.add_argument("--query_docs_eval", type=int, default=1, help="number of query documents (eval)")
    parser.add_argument("--samples_per_ep", type=int, default=2000, help="number of samples per epoch")
    parser.add_argument("--samples_data_train", type=int, default=50000, help="number of training episodes to generate")
    parser.add_argument("--samples_data_dev", type=int, default=500, help="number of dev episodes to generate")
    parser.add_argument("--samples_data_test", type=int, default=10000, help="number of test episodes to generate")
    parser.add_argument("--balancing_train", type=str, default="single", help="balancing (hard, soft, single)")
    parser.add_argument("--balancing_eval", type=str, default="single", help="balancing (hard, soft, single)")
    parser.add_argument("--dataset", type=str, default="FREDo", help="dataset (FREDo, ReFREDo)")
    parser.add_argument("--eval_batch_size", type=int, default=2, help="eval batch size")
    parser.add_argument("--train_batch_size", type=int, default=2, help="training batch size")
    parser.add_argument("--warmup_epochs", type=int, default=1, help="warmup epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-6, help="learning rate")
    parser.add_argument("--loss", type=str, default="atloss", help="loss function")
    parser.add_argument("--ensure_positive", type=bool, default=True, help="ensure positive example query")
    parser.add_argument("--load_checkpoint", type=str, default="checkpoints/FREDo_HQBNxsESIs.pt", help="path to checkpoint")
    parser.add_argument("--project", type=str, default="FREDo", help="project name for wandb")
    parser.add_argument("--random_string", type=str, default=random_string, help="random string for wandb")
    args = parser.parse_args()
    if not debug:
        wandb.init(project=args.project)
        wandb.config.update(args)
        wandb.config.identifier = random_string

    torch.backends.cudnn.enable =True,
    torch.backends.cudnn.benchmark = True
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print('before tokenizer')
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    train_only = False
    markers = args.use_markers
    samples_per_epoch = args.samples_per_ep
    n_epochs = args.num_epochs
    train_batch_size = args.train_batch_size
    warmup_epochs = args.warmup_epochs
    learning_rate = args.learning_rate
    print('before dataloader')
    print(random_string) 
    if args.num_epochs != 0:

        if os.path.exists('data_advance/'+args.dataset+"/"+args.balancing_train+'/training_dataset_{0}doc.pkl'.format(args.support_docs_eval)):
            print('exit train')
            with open('data_advance/'+args.dataset+"/"+args.balancing_train+'/training_dataset_{0}doc.pkl'.format(args.support_docs_eval), 'rb') as f:
                training_episodes = pickle.load(f)
            
        else:
            print('not exit train')
            training_episodes = parse_episodes("data/"+args.dataset+"/train.json", tokenizer, K=args.support_docs_train, n_queries=args.query_docs_train, n_samples=args.samples_data_train, markers=args.use_markers, balancing=args.balancing_train, seed=args.seed_data, ensure_positive=args.ensure_positive, cache="data_cache/"+args.dataset+"/"+args.balancing_train)
            with open('data_advance/'+args.dataset+"/"+args.balancing_train+'/training_dataset_{0}doc.pkl'.format(args.support_docs_eval),'wb') as p:
                pickle.dump(training_episodes,p) 

        if  os.path.exists('data_advance/'+args.dataset+"/"+args.balancing_eval+'/dev_dataset_{0}doc.pkl'.format(args.support_docs_eval)):
            print('exit dev')
            with open('data_advance/'+args.dataset+"/"+args.balancing_eval+'/dev_dataset_{0}doc.pkl'.format(args.support_docs_eval), 'rb') as f:
                dev_episodes = pickle.load(f)
        else:
            print('not exit dev')
            dev_episodes = parse_episodes("data/"+args.dataset+"/dev.json", tokenizer, K=args.support_docs_eval, n_queries=args.query_docs_eval, n_samples=args.samples_data_dev, markers=args.use_markers, balancing=args.balancing_eval, seed=args.seed_data, ensure_positive=args.ensure_positive, cache="data_cache/"+args.dataset+"/"+args.balancing_eval)
            with open('data_advance/'+args.dataset+"/"+args.balancing_eval+'/dev_dataset_{0}doc.pkl'.format(args.support_docs_eval),'wb') as p:
                pickle.dump(dev_episodes,p)
    
    
    
    if os.path.exists('data_advance/'+args.dataset+'/indomain_dataset_{0}doc.pkl'.format(args.support_docs_eval)):
        print('indomain exit')
        with open('data_advance/'+args.dataset+'/indomain_dataset_{0}doc.pkl'.format(args.support_docs_eval), 'rb') as f:
            indomain_test_episodes = pickle.load(f)
    else:
        print('indomain not exit')
        indomain_test_episodes = parse_episodes_from_index("data/"+args.dataset+"/test_docred.json", "data/"+args.dataset+"/test_in_domain_{0}_doc_indices.json".format(args.support_docs_eval), tokenizer, markers=args.use_markers, cache="data_cache/"+args.dataset+"/"+args.balancing_eval)
        with open('data_advance/'+args.dataset+'/indomain_dataset_{0}doc.pkl'.format(args.support_docs_eval),'wb') as p:
            pickle.dump(indomain_test_episodes,p)
            
    if os.path.exists('data_advance/'+args.dataset+'/scierc_dataset_{0}doc.pkl'.format(args.support_docs_eval)):
        print('scierc exit')
        with open('data_advance/'+args.dataset+'/scierc_dataset_{0}doc.pkl'.format(args.support_docs_eval), 'rb') as f:
            scierc_test_episodes = pickle.load(f)       
        
    else:
        print('scierc not exit')
        scierc_test_episodes = parse_episodes_from_index("data/"+args.dataset+"/test_scierc.json", "data/"+args.dataset+"/test_cross_domain_{0}_doc_indices.json".format(args.support_docs_eval), tokenizer, markers=args.use_markers, cache="data_cache/"+args.dataset+"/"+args.balancing_eval)
        with open('data_advance/'+args.dataset+'/scierc_dataset_{0}doc.pkl'.format(args.support_docs_eval),'wb') as p:
            pickle.dump(scierc_test_episodes,p)


    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    g = torch.Generator()
    g.manual_seed(args.seed_data)
    set_seed(args.seed_model)

    if args.num_epochs != 0:
        train_loader = DataLoader(
            training_episodes, 
            batch_size=train_batch_size, 
            shuffle=True, 
            collate_fn=collate_fn_train, num_workers=4, drop_last=True, generator=g)

    else:
        train_loader = []
    if not train_only and args.num_epochs != 0:
        dev_loader = DataLoader(
            dev_episodes, 
            batch_size=args.eval_batch_size, 
            shuffle=False, 
            collate_fn=collate_fn_train, num_workers=4, drop_last=False)

    indomain_test_loader = DataLoader(
            indomain_test_episodes, 
            batch_size=args.eval_batch_size, 
            shuffle=False, 
            collate_fn=collate_fn_train, num_workers=4, drop_last=False)

    scierc_test_loader = DataLoader(
            scierc_test_episodes, 
            batch_size=args.eval_batch_size, 
            shuffle=False, 
            collate_fn=collate_fn_train, num_workers=4, drop_last=False)


    lm_config = AutoConfig.from_pretrained(
        "bert-base-cased",
        num_labels=10,
    )
    lm_model = AutoModel.from_pretrained(
        "bert-base-cased",
        from_tf=False,
        config=lm_config,
    )


    encoder = Encoder(
        config=lm_config,
        model=lm_model, 
        cls_token_id=tokenizer.convert_tokens_to_ids(tokenizer.cls_token), 
        sep_token_id=tokenizer.convert_tokens_to_ids(tokenizer.sep_token),
        markers=markers,
        device=device_debug,
        combination=args.mention_combination,
        gamma_pos=args.gamma_pos, 
        nota_length=args.nota_length,
        dropout=args.dropout,
        support_proto_counts=args.support_proto_counts,
        nota_transform_multi=args.nota_transform_multi,
        )
    encoder.to(device_debug)

    if args.load_checkpoint is not None:
        print(f'loading model from {args.load_checkpoint}')
        encoder.load_state_dict(torch.load(f"{args.load_checkpoint}"))
        total_params = sum(p.numel() for p in encoder.parameters())
        print(f"Total parameters: {total_params}")
    pretrained = encoder.model.parameters()
    pretrained_names = [f'model.{k}' for (k, v) in encoder.model.named_parameters()]
    new_params= [k for k, v in encoder.named_parameters() if k not in pretrained_names]
    optimizer_grouped_parameters = [
    {"params": [p for n, p in encoder.model.named_parameters() if not any(nd in n for nd in new_params)], },
    {"params": [p for n, p in encoder.model.named_parameters() if any(nd in n for nd in new_params)], "lr": 5e-4}
    ]
    if args.single_lr:
        para_opt=encoder.parameters()
    else:
        para_opt=optimizer_grouped_parameters
    optimizer = AdamW(para_opt, lr=learning_rate, eps=1e-6, weight_decay=args.weight_decay)
    scaler = GradScaler()
    num_samples = len(train_loader)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, int(warmup_epochs * samples_per_epoch/train_batch_size), int(samples_per_epoch/train_batch_size*n_epochs))

    step_global = -1

    train_iter = iter(train_loader)
    if args.freelb:
        freelb=FreeLB(args.adv_K, args.adv_lr, args.adv_init_mag, args.adv_max_norm, args.adv_norm_type, base_model='bert')

    best_f1 = 0.0
    f_epoch_lr=open("epoch_lr.txt","w+")
    for i in tqdm(range(n_epochs)):
        true_positives, false_positives, false_negatives = {},{},{}

        encoder.train()
        loss_agg = 0
        count = 0
        with tqdm(range(int(samples_per_epoch/train_batch_size))) as pbar:
            for _ in pbar:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
                step_global += 1
                exemplar_tokens, exemplar_mask, exemplar_positions, exemplar_labels, query_tokens, query_mask, query_positions, query_labels, label_types= batch
                with autocast():
                    try:
                        output, loss = encoder(exemplar_tokens.to(device_debug), exemplar_mask.to(device_debug), exemplar_positions, exemplar_labels, query_tokens.to(device_debug), query_mask.to(device_debug), query_positions, query_labels, label_types)

                    except RuntimeError as exception:
                                raise exception
                for pred, lbls in zip(output, query_labels):
                    for preds, lbs in zip(pred, lbls):
                        for inf in preds:
                            if inf[2] not in true_positives.keys():
                                true_positives[inf[2]] = 0
                                false_positives[inf[2]] = 0
                                false_negatives[inf[2]] = 0

                            if inf in lbs:
                                true_positives[inf[2]] += 1
                            else:
                                false_positives[inf[2]] += 1

                        for label in lbs:
                            if label[2] not in true_positives.keys():
                                true_positives[label[2]] = 0
                                false_positives[label[2]] = 0
                                false_negatives[label[2]] = 0

                            if label not in preds:
                                false_negatives[label[2]] += 1
                count += 1
                loss_agg += loss.item()
                pbar.set_postfix({"Loss":f"{loss_agg/count:.2f}"})
                if not debug:
                    wandb.log({"loss": loss.item()}, step=step_global)
                    wandb.log({"learning_rate": lr_scheduler.get_last_lr()[0]}, step=step_global)
                scaler.scale(loss).backward()
                if args.freelb:
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
                    output_fgm,loss_freelb = freelb.attack(encoder, inputs_freelb,scaler=scaler)
                scaler.unscale_(optimizer)     
                nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                encoder.zero_grad()
                del loss, output
                

        p,r,f = get_f1(true_positives, false_positives, false_negatives)
        p_train, r_train, f1_train = get_f1_macro(true_positives, false_positives, false_negatives, prnt=True)
        if not debug:
            wandb.log({"precision_train": p_train}, step=step_global)
            wandb.log({"recall_train": r_train}, step=step_global)
            wandb.log({"f1_macro_train": f1_train}, step=step_global)
        if not train_only:

            true_positives, false_positives, false_negatives = {},{},{}


            encoder.eval()

            max_memory_dev=0
            with tqdm(dev_loader) as pbar:
                for i,batch in enumerate(pbar):
                    with torch.no_grad():
                        exemplar_tokens, exemplar_mask, exemplar_positions, exemplar_labels, query_tokens, query_mask, query_positions, query_labels, label_types = batch
                        output,loss_dev= encoder(exemplar_tokens.to(device_debug), exemplar_mask.to(device_debug), exemplar_positions, exemplar_labels, query_tokens.to(device_debug), query_mask.to(device_debug), query_positions,  query_labels, label_types)
                    for pred, lbls in zip(output, query_labels):
                        for preds, lbs in zip(pred, lbls):
                            for inf in preds:
                                if inf[2] not in true_positives.keys():
                                    true_positives[inf[2]] = 0
                                    false_positives[inf[2]] = 0
                                    false_negatives[inf[2]] = 0

                                if inf in lbs:
                                    true_positives[inf[2]] += 1
                                else:
                                    false_positives[inf[2]] += 1

                            for label in lbs:
                                if label[2] not in true_positives.keys():
                                    true_positives[label[2]] = 0
                                    false_positives[label[2]] = 0
                                    false_negatives[label[2]] = 0
                                if label not in preds:
                                    false_negatives[label[2]] += 1
                    
                    if not debug:
                        wandb.log({"loss_dev": loss_dev.item()}, step=step_global)
                
            p,r,f = get_f1(true_positives, false_positives, false_negatives)
            p_dev, r_dev, f1_dev = get_f1_macro(true_positives, false_positives, false_negatives, prnt=True)
            if not debug:
                print(i," ",optimizer.state_dict()['param_groups'][0]['lr'],file=f_epoch_lr,flush=True)
                if f1_dev >= best_f1:
                    wandb.log({"best_f1_macro_dev": f1_dev}, step=step_global)
                    best_f1 = f1_dev
                    torch.save(encoder.state_dict(), f"checkpoints/{args.project}_{random_string}.pt")
                else:
                    torch.save(encoder.state_dict(), f"checkpoints/{args.project}_{random_string}_normal.pt")
            if not debug:
                wandb.log({"precision_dev": p_dev}, step=step_global)
                wandb.log({"recall_dev": r_dev}, step=step_global)
                wandb.log({"f1_macro_dev": f1_dev}, step=step_global)
    
    f_epoch_lr.close()
    print(random_string) 
    print("---- INDOMAIN TEST EVAL -----")
    if n_epochs > 0:
        encoder.load_state_dict(torch.load(f"checkpoints/{args.project}_{random_string}.pt"))
    else:
        step_global = 0

    true_positives, false_positives, false_negatives = {},{},{}


    encoder.eval()
    with tqdm(indomain_test_loader) as pbar:
        for batch in pbar:
            with torch.no_grad():
                
                exemplar_tokens, exemplar_mask, exemplar_positions, exemplar_labels, query_tokens, query_mask, query_positions, query_labels, label_types = batch
                
                output= encoder(exemplar_tokens.to(device_debug), exemplar_mask.to(device_debug), exemplar_positions, exemplar_labels, query_tokens.to(device_debug), query_mask.to(device_debug), query_positions, None, label_types)
            for pred, lbls in zip(output, query_labels):
                for preds, lbs in zip(pred, lbls):
                    for inf in preds:
                        if inf[2] not in true_positives.keys():
                            true_positives[inf[2]] = 0
                            false_positives[inf[2]] = 0
                            false_negatives[inf[2]] = 0

                        if inf in lbs:
                            true_positives[inf[2]] += 1
                        else:
                            false_positives[inf[2]] += 1

                    for label in lbs:
                        if label[2] not in true_positives.keys():
                            true_positives[label[2]] = 0
                            false_positives[label[2]] = 0
                            false_negatives[label[2]] = 0

                        if label not in preds:
                            false_negatives[label[2]] += 1

    p,r,f = get_f1(true_positives, false_positives, false_negatives)
    p_dev, r_dev, f1_dev = get_f1_macro(true_positives, false_positives, false_negatives, prnt=True)
    if not debug:
        wandb.log({"precision_test_indomain": p_dev}, step=step_global)
        wandb.log({"recall_test_indomain": r_dev}, step=step_global)
        wandb.log({"f1_macro_test_indomain": f1_dev}, step=step_global)
        wandb.log({"f1_micro_test_indomain": f}, step=step_global)



    print("---- SCIERC TEST EVAL -----")

    true_positives, false_positives, false_negatives = {},{},{}

    with tqdm(scierc_test_loader) as pbar:
        for batch in pbar:
            with torch.no_grad():
                exemplar_tokens, exemplar_mask, exemplar_positions, exemplar_labels, query_tokens, query_mask, query_positions, query_labels, label_types= batch
                output = encoder(exemplar_tokens.to(device_debug), exemplar_mask.to(device_debug), exemplar_positions, exemplar_labels, query_tokens.to(device_debug), query_mask.to(device_debug), query_positions, None, label_types)
            for pred, lbls in zip(output, query_labels):
                for preds, lbs in zip(pred, lbls):
                    for inf in preds:
                        if inf[2] not in true_positives.keys():
                            true_positives[inf[2]] = 0
                            false_positives[inf[2]] = 0
                            false_negatives[inf[2]] = 0

                        if inf in lbs:
                            true_positives[inf[2]] += 1
                        else:
                            false_positives[inf[2]] += 1

                    for label in lbs:
                        if label[2] not in true_positives.keys():
                            true_positives[label[2]] = 0
                            false_positives[label[2]] = 0
                            false_negatives[label[2]] = 0

                        if label not in preds:
                            false_negatives[label[2]] += 1

    p,r,f = get_f1(true_positives, false_positives, false_negatives)
    p_dev, r_dev, f1_dev = get_f1_macro(true_positives, false_positives, false_negatives, prnt=True)
    if not debug:
        wandb.log({"precision_test_scierc": p_dev}, step=step_global)
        wandb.log({"recall_test_scierc": r_dev}, step=step_global)
        wandb.log({"f1_macro_test_scierc": f1_dev}, step=step_global)
        wandb.log({"f1_micro_test_scierc": f}, step=step_global)