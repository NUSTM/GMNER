import sys
sys.path.append('../')
import os
if 'p' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['p']
    # os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import warnings
                                                 
warnings.filterwarnings('ignore')


from model.data_pipe import BartNERPipe
from model.bart_multi_concat import BartSeq2SeqModel  
from model.generater_multi_concat import SequenceGeneratorModel     
from model.metrics import Seq2SeqSpanMetric 
from model.losses import get_loss

import fitlog
import datetime
from fastNLP import Trainer

from torch import optim
from fastNLP import BucketSampler, GradientClipCallback, cache_results, EarlyStopCallback, SequentialSampler

from model.callbacks import WarmupCallback
from fastNLP.core.sampler import SortedSampler
from fastNLP.core.sampler import  ConstTokenNumSampler
from model.callbacks import FitlogCallback
from fastNLP import DataSetIter
from tqdm import tqdm, trange
from fastNLP.core.utils import _move_dict_value_to_device
import random

fitlog.debug()
fitlog.set_log_dir('logs')




import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--bart_name', default='facebook/bart-large', type=str)
parser.add_argument('--datapath', default='./Twitter_GMNER/txt/', type=str)
parser.add_argument('--image_feature_path',default='./data/Twitter_GMNER_vinvl', type=str)
parser.add_argument('--image_annotation_path',default='./Twitter_GMNER/xml/', type=str)
parser.add_argument('--region_loss_ratio',default='1.0', type=float)
parser.add_argument('--box_num',default='16', type=int)
parser.add_argument('--normalize',default=False, action = "store_true")
parser.add_argument('--use_kl',default=False,action ="store_true")
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--n_epochs', default=30, type=int)
parser.add_argument('--max_len', default=30, type=int)
parser.add_argument('--batch_size',default=16,type=int)
parser.add_argument('--seed',default=42,type=int)
parser.add_argument("--save_model",default=0,type=int)
parser.add_argument("--save_path",default='save_models/best',type=str)
parser.add_argument("--log",default='./logs',type=str)
args= parser.parse_args()


dataset_name = 'twitter-ner'
args.length_penalty = 1



args.target_type = 'word'
args.schedule = 'linear'
args.decoder_type = 'avg_feature'
args.num_beams = 1   
args.use_encoder_mlp = 1
args.warmup_ratio = 0.01
eval_start_epoch = 0


if 'twitter' in dataset_name:  
    max_len, max_len_a = args.max_len, 0.6
else:
    print("Error dataset_name!")


if isinstance(args.decoder_type, str) and args.decoder_type.lower() == 'none':
    args.decoder_type = None
demo = False

def get_data():

    pipe = BartNERPipe(image_feature_path=args.image_feature_path, 
                       image_annotation_path=args.image_annotation_path,
                       max_bbox =args.box_num,
                       normalize=args.normalize,
                       tokenizer=args.bart_name, 
                       target_type=args.target_type)
    if dataset_name == 'twitter-ner': 
        paths ={
            'train': os.path.join(args.datapath,'train.txt'),
            'dev': os.path.join(args.datapath,'dev.txt'),
            'test': os.path.join(args.datapath,'test.txt') }
        data_bundle = pipe.process_from_file(paths, demo=demo)
        
    return data_bundle, pipe.tokenizer, pipe.mapping2id

data_bundle, tokenizer, mapping2id = get_data()

print(f'max_len_a:{max_len_a}, max_len:{max_len}')

print(data_bundle)
print("The number of tokens in tokenizer ", len(tokenizer.decoder))  

bos_token_id = 0
eos_token_id = 1
label_ids = list(mapping2id.values())


model = BartSeq2SeqModel.build_model(args.bart_name, tokenizer, label_ids=label_ids, decoder_type=args.decoder_type,
                                     use_encoder_mlp=args.use_encoder_mlp,box_num = args.box_num)


vocab_size = len(tokenizer)

model = SequenceGeneratorModel(model, bos_token_id=bos_token_id,
                               eos_token_id=eos_token_id, 
                               max_length=max_len, max_len_a=max_len_a,num_beams=args.num_beams, do_sample=False,
                               repetition_penalty=1, length_penalty=args.length_penalty, pad_token_id=eos_token_id,
                               restricter=None, top_k = 1
                               )

## parameter scale
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameter: %.2fM" % (total/1e6))
##

import torch
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

torch.manual_seed(args.seed)


parameters =[]
params = {'lr':args.lr}
params['params'] = [param for name, param in model.named_parameters() ]
parameters.append(params)

optimizer = optim.AdamW(parameters)


metric = Seq2SeqSpanMetric(eos_token_id, num_labels=len(label_ids), region_num =args.box_num, target_type=args.target_type,print_mode = False )

train_dataset = data_bundle.get_dataset('train')
eval_dataset = data_bundle.get_dataset('dev')
test_dataset = data_bundle.get_dataset('test')
print(train_dataset[:3])

device = torch.device(device)
model.to(device)

def Training(args, train_idx, train_data, model, device, optimizer):
    
    train_sampler = BucketSampler(seq_len_field_name='src_seq_len',batch_size=args.batch_size)   # 带Bucket的 Random Sampler. 可以随机地取出长度相似的元素
    train_data_iterator = DataSetIter(train_data, batch_size=args.batch_size, sampler=train_sampler)
    
    train_loss = 0.
    train_region_loss = 0.
    # for batch_x, batch_y in tqdm(train_data_iterator, total=len(train_data_iterator)):
    for batch_x, batch_y in (train_data_iterator):
        _move_dict_value_to_device(batch_x, batch_y, device=device)
        src_tokens = batch_x['src_tokens']
        image_feature = batch_x['image_feature']
        tgt_tokens = batch_x['tgt_tokens']
        src_seq_len = batch_x['src_seq_len']
        tgt_seq_len = batch_x['tgt_seq_len']
        first = batch_x['first']
        region_label = batch_y['region_label']


        results = model(src_tokens,image_feature, tgt_tokens, src_seq_len=src_seq_len, tgt_seq_len=tgt_seq_len, first=first)
        pred, region_pred = results['pred'],results['region_pred']   ## logits:(bsz,tgt_len,class+max_len)  region_logits:(??,8)
        
        loss, region_loss = get_loss(tgt_tokens, tgt_seq_len, pred, region_pred,region_label,use_kl=args.use_kl)
        
        train_loss += loss.item()
        train_region_loss += region_loss.item()

        all_loss = loss + args.region_loss_ratio * region_loss
    
        all_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print("train_loss: %f"%(train_loss))
    print("train_region_loss: %f"%(train_region_loss))
    return train_loss, train_region_loss

def Inference(args,eval_data, model, device, metric):
    data_iterator = DataSetIter(eval_data, batch_size=args.batch_size * 2, sampler=SequentialSampler())
    # for batch_x, batch_y in tqdm(data_iterator, total=len(data_iterator)):
    for batch_x, batch_y in (data_iterator):
        _move_dict_value_to_device(batch_x, batch_y, device=device)
        src_tokens = batch_x['src_tokens']
        image_feature = batch_x['image_feature']
        tgt_tokens = batch_x['tgt_tokens']
        src_seq_len = batch_x['src_seq_len']
        tgt_seq_len = batch_x['tgt_seq_len']
        first = batch_x['first']
        region_label = batch_y['region_label']
        target_span = batch_y['target_span']
        cover_flag = batch_y['cover_flag']

        results = model.predict(src_tokens,image_feature, src_seq_len=src_seq_len, first=first)
        
        pred,region_pred = results['pred'],results['region_pred']   ## logits:(bsz,tgt_len,class+max_len)  region_logits:(??,8)
        
        metric.evaluate(target_span, pred, tgt_tokens, region_pred,region_label,cover_flag)
    res = metric.get_metric()  ## {'f': 20.0, 'rec': 16.39, 'pre': 25.64, 'em': 0.125, 'uc': 0}
    return res


def Predict(args,eval_data, model, device, metric,tokenizer,ids2label):
    data_iterator = DataSetIter(eval_data, batch_size=args.batch_size * 2, sampler=SequentialSampler())
    # for batch_x, batch_y in tqdm(data_iterator, total=len(data_iterator)):
    with open (args.pred_output_file,'w') as fw:
        for batch_x, batch_y in (data_iterator):
            _move_dict_value_to_device(batch_x, batch_y, device=device)
            src_tokens = batch_x['src_tokens']
            image_feature = batch_x['image_feature']
            tgt_tokens = batch_x['tgt_tokens']
            src_seq_len = batch_x['src_seq_len']
            tgt_seq_len = batch_x['tgt_seq_len']
            first = batch_x['first']
            region_label = batch_y['region_label']
            target_span = batch_y['target_span']
            cover_flag = batch_y['cover_flag']

            results = model.predict(src_tokens,image_feature, src_seq_len=src_seq_len, first=first)
            
            pred,region_pred = results['pred'],results['region_pred']   ## logits:(bsz,tgt_len,class+max_len)  region_logits:(??,8)
            
            pred_pairs, target_pairs = metric.evaluate(target_span, pred, tgt_tokens, region_pred,region_label,cover_flag,predict_mode=True)
            
            raw_words = batch_y['raw_words']
            word_start_index = 8 ## 2 + 2 +4
            assert len(pred_pairs) == len(target_pairs)
            for i in range(len(pred_pairs)):
                cur_src_token = src_tokens[i].cpu().numpy().tolist()
                fw.write(' '.join(raw_words[i])+'\n')
                fw.write('Pred: ')
                for k,v in pred_pairs[i].items():
                    entity_span_ind_list =[]
                    for kk in k:
                        entity_span_ind_list.append(cur_src_token[kk-word_start_index])
                    entity_span = tokenizer.decode(entity_span_ind_list)
                    
                    region_pred, entity_type_ind = v
                    entity_type = ids2label[entity_type_ind[0]]
                    
                    fw.write('('+entity_span+' , '+ str(region_pred)+' , '+entity_type+' ) ')
                fw.write('\n')
                fw.write(' GT : ')
                for k,v in target_pairs[i].items():
                    entity_span_ind_list =[]
                    for kk in k:
                        entity_span_ind_list.append(cur_src_token[kk-word_start_index])
                    entity_span = tokenizer.decode(entity_span_ind_list)
        
                    region_pred, entity_type_ind = v
                    entity_type = ids2label[entity_type_ind[0]]

                    fw.write('('+entity_span+' , '+ str(region_pred)+' , '+entity_type+' ) ')
                fw.write('\n\n')
        res = metric.get_metric()  
        fw.write(str(res))
    return res



max_dev_f = 0.
max_test_f = 0.
best_dev = {}
best_test = {}
best_dev_corresponding_test = {}

for train_idx in range(args.n_epochs):
    print("-"*12+"Epoch: "+str(train_idx)+"-"*12)

    model.train()
    train_loss, train_region_loss = Training(args,train_idx=train_idx,train_data=train_dataset, model=model, device=device,
                                                optimizer=optimizer)
    

    model.eval()
    dev_res = Inference(args,eval_data=eval_dataset, model=model, device=device, metric = metric)
    dev_f = dev_res['f']
    print("dev: "+str(dev_res))

   
    test_res = Inference(args,eval_data=test_dataset, model=model, device=device, metric = metric)
    
    
    test_f = test_res['f']
    print("test: "+str(test_res))

    train_res = Inference(args,eval_data=train_dataset, model=model, device=device, metric = metric)
    train_f = train_res['f']
    print("train: "+str(train_res))



    if dev_f >= max_dev_f:
        max_dev_f = dev_f 
        if args.save_model:
            model_to_save = model.module if hasattr(model, 'module') else model  
            torch.save(model_to_save.state_dict(), args.save_path)
        best_dev = dev_res
        best_dev['epoch'] = train_idx
        best_dev_corresponding_test = test_res
        best_dev_corresponding_test['epoch'] = train_idx
        
   
    if test_f >= max_test_f:
        max_test_f = test_f 
        best_test = test_res
        best_test['epoch'] = train_idx

print("                   best_dev: "+str(best_dev))
print("best_dev_corresponding_test: "+str(best_dev_corresponding_test))
print("                  best_test: "+str(best_test))

if args.save_path and args.save_model:
    print("-"*12+'Predict'+'-'*12)
    ids2label = {2+i:l for i,l in enumerate(mapping2id.keys())}

    model_path = args.save_path.rsplit('/')
    args.pred_output_file = '/'.join(model_path[:-1])+'/pred_'+model_path[-1]+'.txt'

    model.load_state_dict(torch.load(args.save_path))
    model.to(device)

    print(test_dataset[:3])
    test_dataset.set_target('raw_words', 'raw_target')


    model.eval()
    test_res = Predict(args,eval_data=test_dataset, model=model, device=device, metric = metric,tokenizer=tokenizer,ids2label=ids2label)
    test_f = test_res['f']
    print("test: "+str(test_res))