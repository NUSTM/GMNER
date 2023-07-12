
# from fastNLP import LossBase
import torch.nn.functional as F
from fastNLP import seq_len_to_mask
import torch


# class Seq2SeqLoss(LossBase):
#     def __init__(self):
#         super().__init__()

def get_loss(tgt_tokens, tgt_seq_len, pred, region_pred,region_label,use_kl = True):
    
    tgt_seq_len = tgt_seq_len - 1  ### 不算开始符号
    mask = seq_len_to_mask(tgt_seq_len, max_len=tgt_tokens.size(1) - 1).eq(0)
    tgt_tokens = tgt_tokens[:, 1:].masked_fill(mask, -100)   ### 处理之后没有开始0，[  57,   58,   59,   60,    2,    1, -100, -100]
    loss = F.cross_entropy(target=tgt_tokens, input=pred.transpose(1, 2))

   
    
    region_mask = region_label[:,:,:-1].sum(dim=-1).gt(0)   ## only for related 

    if region_pred is not None and region_mask.sum()!=0:   
        if use_kl:
            bbox_num = region_pred.size(-1)
           
            region_loss = F.kl_div(input = F.log_softmax(region_pred), target = region_label[region_mask][:,:-1],reduction= 'batchmean') 
        
        ## BCE
        else:
           
            region_label = region_label[region_mask][:,:-1]
            pos_tag = region_label.new_full(region_label.size(),fill_value = 1.)
            neg_tag = region_label.new_full(region_label.size(),fill_value = 0.)
            BCE_target = torch.where(region_label > 0,pos_tag,neg_tag)
            bbox_num = region_pred.size(-1)
            sample_weight = region_pred.new_full((bbox_num,),fill_value=1.)
            region_loss = F.binary_cross_entropy_with_logits(region_pred, target=BCE_target ,weight =  sample_weight)
    else:
        region_loss = torch.tensor(0.,requires_grad=True).to(loss.device)
    
    return loss , region_loss

