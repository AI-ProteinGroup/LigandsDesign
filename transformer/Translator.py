''' This module will handle the text generation with beam search. '''
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import utils
from transformer.Models import get_pad_mask, get_subsequent_mask, PositionalEncoding
import random

class Translator(nn.Module):
    ''' Load a trained model and translate in beam search fashion. '''

    def __init__(
            self, model, beam_size, max_seq_len,
            src_pad_idx, trg_pad_idx, trg_bos_idx, trg_eos_idx):
        

        super(Translator, self).__init__()

        self.alpha = 0.7
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len
        self.src_pad_idx = src_pad_idx
        self.trg_bos_idx = trg_bos_idx
        self.trg_eos_idx = trg_eos_idx

        self.model = model
        self.model.eval()

        self.register_buffer('init_seq', torch.LongTensor([[trg_bos_idx]]))
        self.register_buffer(
            'blank_seqs', 
            torch.full((beam_size, max_seq_len), trg_pad_idx, dtype=torch.long))

        self.register_buffer(
            'blank_seq', 
            torch.full((1, max_seq_len), trg_pad_idx, dtype=torch.long))
        self.blank_seqs[:, 0] = self.trg_bos_idx
        self.blank_seq[:, 0] = self.trg_bos_idx
        self.register_buffer(
            'len_map', 
            torch.arange(1, max_seq_len + 1, dtype=torch.long).unsqueeze(0))
        
        self.model.decoder.position_enc = PositionalEncoding(self.model.d_model, n_position=max_seq_len)

    def _model_decode(self, trg_seq, enc_output, src_mask):
        trg_mask = get_subsequent_mask(trg_seq)
        dec_output = self.model.decoder(trg_seq, trg_mask, enc_output, src_mask)  # beam_size x len x model_d
        return F.softmax(self.model.trg_word_prj(dec_output), dim=-1)  # beam_size x len x d_vocab

    def _get_init_state(self, src_seq, src_mask):
        beam_size = self.beam_size

        enc_output = self.model.encoder(src_seq, src_mask)   # 1 x len x embedding_d
        dec_output = self._model_decode(self.init_seq, enc_output, src_mask)   # 1 x 1 x d_vocab
        best_k_probs, best_k_idx = dec_output[:, -1, :].topk(beam_size)  

        scores = torch.log(best_k_probs).view(beam_size)
        gen_seq = self.blank_seqs.clone().detach()  # beam_size x maxlen
        gen_seq[:, 1] = best_k_idx[0]  # beam_size x maxlen   
        enc_output = enc_output.repeat(beam_size, 1, 1) # beam_size x src_len x embedding
        return enc_output, gen_seq, scores

    def _get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step):
        assert len(scores.size()) == 1
        
        beam_size = self.beam_size

        # Get k candidates for each beam, k^2 candidates in total.
        best_k2_probs, best_k2_idx = dec_output[:, -1, :].topk(beam_size)

        # Include the previous scores.
        scores = torch.log(best_k2_probs).view(beam_size, -1) + scores.view(beam_size, 1)

        # Get the best k candidates from k^2 candidates.
        scores, best_k_idx_in_k2 = scores.view(-1).topk(beam_size)
 
        # Get the corresponding positions of the best k candidiates.
        best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 % beam_size
        best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]

        # Copy the corresponding previous tokens.
        gen_seq[:, :step] = gen_seq[best_k_r_idxs, :step]
        # Set the best tokens in this beam search step
        gen_seq[:, step] = best_k_idx

        return gen_seq, scores

    def translate_sentence_with_beam_search(self, src_seq):
        # Only accept batch size equals to 1 in this function.
        # TODO: expand to batch operation.
        assert src_seq.size(0) == 1

        src_pad_idx, trg_eos_idx = self.src_pad_idx, self.trg_eos_idx 
        max_seq_len, beam_size, alpha = self.max_seq_len, self.beam_size, self.alpha 

        with torch.no_grad():
            src_mask = get_pad_mask(src_seq, src_pad_idx)     # 1 x len
            enc_output, gen_seq, scores = self._get_init_state(src_seq, src_mask)
            # enc_output       beam_size x src_len x embedding
            # gen_seq          beam_size x maxlen
            # score            beam_size

            ans_idx = 0   # default
            '''
            这一块有个问题，那就是beam_search时，如果有句子已经有eos了却不会终止。任务会等到每个句子都有eos才终止��1�7
            那么已有eos的句子仍然会不断生成新句子��1�7          
            '''
            for step in range(2, max_seq_len):    # decode up to max length
                dec_output = self._model_decode(gen_seq[:, :step], enc_output, src_mask) # beam_size x len x d_vocab
                gen_seq, scores = self._get_the_best_score_and_idx(gen_seq, dec_output, scores, step)

                # Check if all path finished
                # -- locate the eos in the generated sequences
                eos_locs = gen_seq == trg_eos_idx   
                # -- replace the eos with its position for the length penalty use
                seq_lens, _ = self.len_map.masked_fill(~eos_locs, max_seq_len).min(1)
                # -- check if all beams contain eos
                if (eos_locs.sum(1) > 0).sum(0).item() == beam_size:
                    # TODO: Try different terminate conditions.
                    _, ans_idx = scores.div(seq_lens.float() ** alpha).max(0)
                    ans_idx = ans_idx.item()
                    break
        return gen_seq[ans_idx][:seq_lens[ans_idx]].tolist()

    def translate_sentence(self, src_seq, randomness = 0):
        assert src_seq.size(0) == 1
        src_pad_idx, trg_eos_idx = self.src_pad_idx, self.trg_eos_idx 
        max_seq_len  = self.max_seq_len
        with torch.no_grad():
            src_mask = get_pad_mask(src_seq, src_pad_idx).view(1, -1)      # 1 x len
            enc_output = self.model.encoder(src_seq, src_mask)         # 1 x len x embedding_d
            gen_seq = self.blank_seq.clone().detach()  # 1 x maxlen

            for step in range(1, max_seq_len):    # decode up to max length
                dec_output = self._model_decode(gen_seq[:, :step], enc_output, src_mask) # 1 x len x d_vocab
                if random.random() <= randomness:
                    action = torch.multinomial(dec_output[:,-1,:].view(-1), 1)
                else:
                    _,action = dec_output[:,-1,:].view(-1).topk(1)
                
                gen_seq[:,step] = action
                if action == trg_eos_idx:
                    return gen_seq[0][:step].tolist()
            return gen_seq[0][:].tolist()

class ProtransTranslator(nn.Module):
    ''' Load a trained model and translate in beam search fashion. '''

    def __init__(
            self, model, beam_size, max_seq_len,
            trg_pad_idx, trg_bos_idx, trg_eos_idx):
        

        super(ProtransTranslator, self).__init__()

        self.alpha = 0.7
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len
        self.trg_bos_idx = trg_bos_idx
        self.trg_eos_idx = trg_eos_idx

        self.model = model
        self.model.eval()

        self.register_buffer('init_seq', torch.LongTensor([[trg_bos_idx]]))
        self.register_buffer(
            'blank_seqs', 
            torch.full((beam_size, max_seq_len), trg_pad_idx, dtype=torch.long))

        self.register_buffer(
            'blank_seq', 
            torch.full((1, max_seq_len), trg_pad_idx, dtype=torch.long))
        self.blank_seqs[:, 0] = self.trg_bos_idx
        self.blank_seq[:, 0] = self.trg_bos_idx
        self.register_buffer(
            'len_map', 
            torch.arange(1, max_seq_len + 1, dtype=torch.long).unsqueeze(0))
        
        self.model.decoder.position_enc = PositionalEncoding(self.model.d_model, n_position=max_seq_len)

    def _model_decode(self, trg_seq, enc_output, src_mask):
        trg_mask = get_subsequent_mask(trg_seq)
        dec_output = self.model.decoder(trg_seq, trg_mask, enc_output, src_mask)  # beam_size x len x model_d
        return F.softmax(self.model.trg_word_prj(dec_output), dim=-1)  # beam_size x len x d_vocab

    def translate_sentence(self, src_seq, src_mask, randomness = 0):
        assert src_seq.size(0) == 1
        
        trg_eos_idx =  self.trg_eos_idx 
        max_seq_len  = self.max_seq_len

        with torch.no_grad():
            enc_output = self.model.bert(input_ids=src_seq,attention_mask=src_mask)[0]
            gen_seq = self.blank_seq.clone().detach()  # 1 x maxlen

            for step in range(1, max_seq_len):    # decode up to max length
                dec_output = self._model_decode(gen_seq[:, :step], enc_output, src_mask) # 1 x len x d_vocab
                if random.random() <= randomness:
                    action = torch.multinomial(dec_output[:,-1,:].view(-1), 1)
                else:
                    _,action = dec_output[:,-1,:].view(-1).topk(1)
                
                gen_seq[:,step] = action
                if action == trg_eos_idx:
                    return gen_seq[0][:step].tolist()
            return gen_seq[0][:].tolist()