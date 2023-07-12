import torch
from .modeing_bart_multi_concat import BartEncoder, BartDecoder, BartModel
from transformers import BartTokenizer
from fastNLP import seq_len_to_mask
from fastNLP.modules import Seq2SeqEncoder, Seq2SeqDecoder, State
import torch.nn.functional as F
from fastNLP.models import Seq2SeqModel
from torch import nn
import math



def mask_iamge(image_feature):
    mask = image_feature.sum(dim=-1).gt(0)
    return mask

class FBartEncoder(Seq2SeqEncoder):
    def __init__(self, encoder):
        super().__init__()
        assert isinstance(encoder, BartEncoder)
        self.bart_encoder = encoder

    def forward(self, src_tokens, image_feature, src_seq_len):
        mask = seq_len_to_mask(src_seq_len, max_len=src_tokens.size(1))
        image_mask = mask_iamge(image_feature)    

        img_feat_, dict = self.bart_encoder(input_ids=src_tokens, image_feature=image_feature, attention_mask=mask, image_mask = image_mask, return_dict=True,
                                 output_hidden_states=True)  # last_hidden_state: tensor(bsz, max_len, 768),  hidden_states: tuple((baz, max_len, 768)),  attentions
        encoder_outputs = dict.last_hidden_state
        hidden_states = dict.hidden_states
        multi_modal_mask = torch.cat((image_mask,mask),dim=-1)
        return img_feat_, encoder_outputs, multi_modal_mask, hidden_states


class FBartDecoder(Seq2SeqDecoder):
    def __init__(self, decoder, pad_token_id, label_ids, use_encoder_mlp=True):
        super().__init__()
        assert isinstance(decoder, BartDecoder)
        self.decoder = decoder
        causal_mask = torch.zeros(512, 512).fill_(float('-inf'))
        causal_mask = causal_mask.triu(diagonal=1)
        self.register_buffer('causal_masks', causal_mask.float())
        self.pad_token_id = pad_token_id
        self.label_start_id = min(label_ids)
        self.label_end_id = max(label_ids)+1
        mapping = torch.LongTensor([0, 2]+label_ids)
        self.register_buffer('mapping', mapping)
        self.src_start_index = len(mapping)  # 加上一个
        hidden_size = decoder.embed_tokens.weight.size(1)
        if use_encoder_mlp:
            self.encoder_mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                             nn.Dropout(0.3),
                                             nn.ReLU(),
                                             nn.Linear(hidden_size, hidden_size))

    def forward(self, tokens, state):
        # bsz, max_len = tokens.size()
        encoder_outputs = state.encoder_output
        encoder_pad_mask = state.encoder_mask

        first = state.first

        # tokens之后的0全是padding，因为1是eos, 在pipe中规定的
        cumsum = tokens.eq(1).flip(dims=[1]).cumsum(dim=-1)
        tgt_pad_mask = cumsum.flip(dims=[1]).ne(cumsum[:, -1:])

        # 把输入做一下映射
        mapping_token_mask = tokens.lt(self.src_start_index)  # 为1的地方应该从mapping中取index
        
        mapped_tokens = tokens.masked_fill(tokens.ge(self.src_start_index), 0)
        tag_mapped_tokens = self.mapping[mapped_tokens]

        src_tokens_index = tokens - self.src_start_index # bsz x num_src_token
        src_tokens_index = src_tokens_index.masked_fill(src_tokens_index.lt(0), 0)
        src_tokens = state.src_tokens
        if first is not None:
            src_tokens = src_tokens.gather(index=first, dim=1)
        word_mapped_tokens = src_tokens.gather(index=src_tokens_index, dim=1)

        tokens = torch.where(mapping_token_mask, tag_mapped_tokens, word_mapped_tokens)
        tokens = tokens.masked_fill(tgt_pad_mask, self.pad_token_id)

        if self.training:
            tokens = tokens[:, :-1]
            decoder_pad_mask = tokens.eq(self.pad_token_id)  # decoder需要让pad位置为1
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=decoder_pad_mask,
                                decoder_causal_mask=self.causal_masks[:tokens.size(1), :tokens.size(1)],
                                return_dict=True)
        else:
            past_key_values = state.past_key_values
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=None,
                                decoder_causal_mask=None,
                                past_key_values=past_key_values,
                                use_cache=True,
                                return_dict=True)
        hidden_state = dict.last_hidden_state  # bsz x max_len x hidden_size
        ## 比CaGFBartDecoder 少一个dropout
        if not self.training:
            state.past_key_values = dict.past_key_values

        logits = hidden_state.new_full((hidden_state.size(0), hidden_state.size(1), self.src_start_index+src_tokens.size(-1)),
                                       fill_value=-1e24)

        # 首先计算的是
        eos_scores = F.linear(hidden_state, self.decoder.embed_tokens.weight[2:3])  # bsz x max_len x 1
        tag_scores = F.linear(hidden_state, self.decoder.embed_tokens.weight[self.label_start_id:self.label_end_id])  # bsz x max_len x num_class

        # bsz x max_word_len x hidden_size
        src_outputs = state.encoder_output

        if hasattr(self, 'encoder_mlp'):
            src_outputs = self.encoder_mlp(src_outputs)

        if first is not None:
            mask = first.eq(0)  # bsz x 1 x max_word_len, 为1的地方是padding
            src_outputs = src_outputs.gather(index=first.unsqueeze(2).repeat(1, 1, src_outputs.size(-1)), dim=1)
        else:
            mask = state.encoder_mask.eq(0)

        mask = mask.unsqueeze(1).__or__(src_tokens.eq(2).cumsum(dim=1).ge(1).unsqueeze(1))
        # 与CaGFBartDecoder不同，此处仅涉及 encoder输出和decoder输出，而CaGFBartDecoder将输入和encoder的输出平均
        word_scores = torch.einsum('blh,bnh->bln', hidden_state, src_outputs)  # bsz x max_len x max_word_len
        word_scores = word_scores.masked_fill(mask, -1e32)

        logits[:, :, 1:2] = eos_scores
        logits[:, :, 2:self.src_start_index] = tag_scores
        logits[:, :, self.src_start_index:] = word_scores

        return logits

    def decode(self,img_feat_,  tokens, state):
        # return self(tokens, state)[:, -1]
        return self(img_feat_, tokens, state)


class CaGFBartDecoder(FBartDecoder):
    def __init__(self, decoder, pad_token_id, label_ids, avg_feature=True, use_encoder_mlp=False,box_num = 36):
        super().__init__(decoder, pad_token_id, label_ids, use_encoder_mlp=use_encoder_mlp)
        self.avg_feature = avg_feature  # 如果是avg_feature就是先把token embed和对应的encoder output平均，
        self.dropout_layer = nn.Dropout(0.3)
        self.box_num = box_num
        hidden_size = decoder.embed_tokens.weight.size(1)
        self.region_select = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                             nn.Dropout(0.3),
                                             nn.ReLU(),
                                             nn.Linear(hidden_size,self.box_num))

    def forward(self, img_feat_, tokens, state):  
        
        bsz, max_len = tokens.size()
        encoder_outputs = state.encoder_output  
        encoder_pad_mask = state.encoder_mask 
        
        first = state.first  # 原始sentence内部是index，之外padding部分是0
        target = tokens
        
        ## tokens to tokenize-id
        # mask target
        cumsum = tokens.eq(1).flip(dims=[1]).cumsum(dim=-1)  
        tgt_pad_mask = cumsum.flip(dims=[1]).ne(cumsum[:, -1:]) 
        # 把输入(即target)做一下映射，变成【encoder】的embed的id
        mapping_token_mask = tokens.lt(self.src_start_index)  
        # 映射类别
        mapped_tokens = tokens.masked_fill(tokens.ge(self.src_start_index), 0)  
        tag_mapped_tokens = self.mapping[mapped_tokens]    # 映射特殊字符
        # 映射token
        src_tokens_index = tokens - self.src_start_index  # 还原index，因为tokens的index从标签类别之后开始
        src_tokens_index = src_tokens_index.masked_fill(src_tokens_index.lt(0), 0)  # 不是token的部分置零 
        src_tokens = state.src_tokens  
        if first is not None:
            src_tokens = src_tokens.gather(index=first, dim=1)  # sentence部分正常取，pad部分取“0”  
        word_mapped_tokens = src_tokens.gather(index=src_tokens_index, dim=1) 
        # 两个映射组合
        tokens = torch.where(mapping_token_mask, tag_mapped_tokens, word_mapped_tokens)  
        tokens = tokens.masked_fill(tgt_pad_mask, self.pad_token_id)  
        #### 以上在准备decoder的input_ids, 将 index, 类别 表示的 taget 转换为 encoder 能读懂的 tokenize id 
        
        if self.training:
            tokens = tokens[:, :-1]  
            decoder_pad_mask = tokens.eq(self.pad_token_id)  # decoder需要让pad位置为1   
            
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,  # (bsz, max_len, 768)
                                encoder_padding_mask=encoder_pad_mask,  
                                decoder_padding_mask=decoder_pad_mask,  # (bsz, max_target-1)
                                decoder_causal_mask=self.causal_masks[:tokens.size(1), :tokens.size(1)], 
                                return_dict=True)  # BaseModelOutputWithPast类，包括last_hidden_state, past_key_values, hidden_states, attentions
        else:
            past_key_values = state.past_key_values
            try:
                dict = self.decoder(input_ids=tokens,
                                    encoder_hidden_states=encoder_outputs,
                                    encoder_padding_mask=encoder_pad_mask,
                                    decoder_padding_mask=None,
                                    decoder_causal_mask=None,
                                    past_key_values=past_key_values,
                                    use_cache=True,
                                    return_dict=True)
            except:
                import pdb;pdb.set_trace()
        hidden_state = dict.last_hidden_state  # bsz x target_len x hidden_size
        hidden_state = self.dropout_layer(hidden_state)
        if not self.training:
            state.past_key_values = dict.past_key_values

        logits = hidden_state.new_full((hidden_state.size(0), hidden_state.size(1), self.src_start_index+src_tokens.size(-1)), fill_value=-1e24)   # (nsz, max_target， 54 + max_len)
        
        
        eos_scores = F.linear(hidden_state, self.dropout_layer(self.decoder.embed_tokens.weight[2:3]))  # (bsz, max_target, 768) x (768 ,1) -> (bsz, max_target, 1)
        tag_scores = F.linear(hidden_state, self.dropout_layer(self.decoder.embed_tokens.weight[self.label_start_id:self.label_end_id]))  # bsz x max_len x num_class                                                                 
        

       
        src_outputs = state.encoder_output[:,self.box_num:,:]  
        src_img_outputs = state.encoder_output[:,:self.box_num,:]

        if hasattr(self, 'encoder_mlp'):
            src_outputs = self.encoder_mlp(src_outputs)
            src_img_outputs = self.encoder_mlp(src_img_outputs)

        if first is not None:
            mask = first.eq(0)  # bsz x 1 x max_word_len 
            src_outputs = src_outputs.gather(index=first.unsqueeze(2).repeat(1, 1, src_outputs.size(-1)), dim=1) # (bsz, max_len, 768)  # 取sentence内的encoder_output
        else:
            print("CaGFBartDecoder: first is None !")
            import pdb;pdb.set_trace()
        mask = mask.unsqueeze(1)

        input_embed = self.dropout_layer(self.decoder.embed_tokens(src_tokens))  # bsz x max_word_len x hidden_size
        input_img_embed = self.dropout_layer(img_feat_)  # bsz, box_num, hidden_size

        if self.avg_feature:  # 先把feature合并一下
            src_outputs = (src_outputs + input_embed)/2   
            src_img_outputs = (src_img_outputs + input_img_embed) /2
        word_scores = torch.einsum('blh,bnh->bln', hidden_state, src_outputs) 
        img_scores = torch.einsum('blh,bnh->bln', hidden_state, src_img_outputs)
        if not self.avg_feature:
            gen_scores = torch.einsum('blh,bnh->bln', hidden_state, input_embed)  
            word_scores = (gen_scores + word_scores)/2
            gen_img_scores = torch.einsum('blh,bnh->bln', hidden_state, input_img_embed)  
            img_scores = (gen_img_scores + img_scores)/2
        
        mask = mask.__or__(src_tokens.eq(2).cumsum(dim=1).ge(1).unsqueeze(1))  # 2 是结束符
        word_scores = word_scores.masked_fill(mask, -1e32)  

        logits[:, :, 1:2] = eos_scores
        logits[:, :, 2:self.src_start_index] = tag_scores
        logits[:, :, self.src_start_index:] = word_scores   # logits: (bsz, target, 类别数 + max_len)
       
        if self.training:

            region_ind = target[:,:-1].eq(2)   ## bsz, max_len
            img_logits = img_scores[region_ind]  ## ??, box_num
            return logits, img_logits   ## logits:(bsz, target_len, n_class+max_len)  region_pred:(bsz, ??, max_box+1 )
        else:  
            
            logits = logits[:,-1,:] ## logits:(bsz, n_class+max_len)
            img_logits = img_scores[:,-1,:]
            return logits, img_logits





class BartSeq2SeqModel(Seq2SeqModel):
    @classmethod  
    def build_model(cls, bart_model, tokenizer, label_ids, decoder_type=None,
                    use_encoder_mlp=False,box_num = 36):
        model = BartModel.from_pretrained(bart_model)
        num_tokens, _ = model.encoder.embed_tokens.weight.shape
        model.resize_token_embeddings(len(tokenizer.unique_no_split_tokens)+num_tokens)   # 扩充vocab
        encoder = model.encoder
        decoder = model.decoder

        # 将类别（eg: "<<person>>"）添加到decoder原本词表之前，embed使用“类别名”的embed
        _tokenizer = BartTokenizer.from_pretrained(bart_model)   
        for token in tokenizer.unique_no_split_tokens:
            if token[:2] == '<<':  # 特殊字符
                index = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
                if len(index)>1:
                    raise RuntimeError(f"{token} wrong split")   
                else:
                    index = index[0]
                assert index>=num_tokens, (index, num_tokens, token) 
                indexes = _tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(token[2:-2])) 
                embed = model.encoder.embed_tokens.weight.data[indexes[0]]  
                for i in indexes[1:]:
                    embed += model.decoder.embed_tokens.weight.data[i]
                embed /= len(indexes)   
                model.decoder.embed_tokens.weight.data[index] = embed  

        encoder = FBartEncoder(encoder)
        if decoder_type is None:
            decoder = FBartDecoder(decoder, pad_token_id=tokenizer.pad_token_id, label_ids=label_ids, box_num =box_num)  # label_ids是"<<actor>>"在_tokenizer中的id，在原词表之后
        elif decoder_type == 'avg_score':
            decoder = CaGFBartDecoder(decoder, pad_token_id=tokenizer.pad_token_id, label_ids=label_ids,
                                      avg_feature=False, use_encoder_mlp=use_encoder_mlp,box_num =box_num)
        elif decoder_type == 'avg_feature':
            decoder = CaGFBartDecoder(decoder, pad_token_id=tokenizer.pad_token_id, label_ids=label_ids,
                                      avg_feature=True, use_encoder_mlp=use_encoder_mlp,box_num =box_num)
        else:
            raise RuntimeError("Unsupported feature.")

        return cls(encoder=encoder, decoder=decoder)

    def prepare_state(self, src_tokens, image_feature,src_seq_len=None, first=None, tgt_seq_len=None):
        img_feat_, encoder_outputs, encoder_mask, hidden_states = self.encoder(src_tokens,image_feature, src_seq_len)
        src_embed_outputs = hidden_states[0]
        state = BartState(encoder_outputs, encoder_mask, src_tokens, first, src_embed_outputs)
        # BartState 包括: src_tokens, first, src_embed_outputs
        return img_feat_, state

    def forward(self, src_tokens,image_feature, tgt_tokens, src_seq_len, tgt_seq_len, first):
        """

        :param torch.LongTensor src_tokens: source的token
        :param torch.LongTensor tgt_tokens: target的token
        :param torch.LongTensor first: 显示每个, bsz x max_word_len
        :param torch.LongTensor src_seq_len: src的长度
        :param torch.LongTensor tgt_seq_len: target的长度，默认用不上
        :return: {'pred': torch.Tensor}, 其中pred的shape为bsz x max_len x vocab_size
        """
        
        img_feat_, state = self.prepare_state(src_tokens, image_feature,src_seq_len, first, tgt_seq_len)
        decoder_output, region_pred = self.decoder(img_feat_, tgt_tokens, state)  # (bsz, max_target, 95) # 95, 每个预测的token上分 max_len+类别数 类
        if isinstance(decoder_output, torch.Tensor):
            return {'pred': decoder_output,'region_pred':region_pred}
        else:
            raise TypeError(f"Unsupported return type from Decoder:{type(self.decoder)}")



class BartState(State):
    def __init__(self, encoder_output, encoder_mask, src_tokens, first, src_embed_outputs):
        super().__init__(encoder_output, encoder_mask)
        self.past_key_values = None
        self.src_tokens = src_tokens
        self.first = first
        self.src_embed_outputs = src_embed_outputs

    def reorder_state(self, indices: torch.LongTensor):
        super().reorder_state(indices)
        self.src_tokens = self._reorder_state(self.src_tokens, indices)
        if self.first is not None:
            self.first = self._reorder_state(self.first, indices)
        self.src_embed_outputs = self._reorder_state(self.src_embed_outputs, indices)
        if self.past_key_values is not None:
            new = []
            for layer in self.past_key_values:
                new_layer = {}
                for key1 in list(layer.keys()):
                    new_layer_ = {}
                    for key2 in list(layer[key1].keys()):
                        if layer[key1][key2] is not None:
                            layer[key1][key2] = self._reorder_state(layer[key1][key2], indices)
                            # print(key1, key2, layer[key1][key2].shape)
                        new_layer_[key2] = layer[key1][key2]
                    new_layer[key1] = new_layer_
                new.append(new_layer)
            self.past_key_values = new