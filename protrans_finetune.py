'''
This script handles the training process.
'''
import math
import os
import random
import time
import copy
from tqdm import tqdm
import dill as pickle
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from utils import vocabulary, dataset
import transformer.Models as model
import transformer.Optim  as op
from transformers import BertModel, BertTokenizer
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word

def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss

def patch_trg(trg):
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    return trg, gold

def train_epoch(model, tokenizer,  training_data, optimizer, device, smoothing):
    ''' Epoch operation in training phase'''

    model.train()
    total_loss, n_word_total, n_word_correct = 0, 0, 0 
    optimizer.zero_grad()
    desc = '  - (Training)   '
    i = 1
    for batch in tqdm(training_data, mininterval=2, desc=desc, leave=False):
        # prepare data
        src_seqs = batch['proteins'] # batch_size x len

        ids = tokenizer.batch_encode_plus(src_seqs, add_special_tokens=True, padding='longest')
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        trg_seqs, gold = map(lambda x: x.to(device), patch_trg(batch['drugs'])) # batch_size x len      1 x (batch_size x len)

        # forward
        pred = model(input_ids, trg_seqs, attention_mask)
        
        # backward and update parameters
        loss, n_correct, n_word = cal_performance(pred, gold, 0, smoothing=smoothing) 
        loss.backward()

        if i % arg["step_batch"] == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # note keeping
        n_word_total += n_word
        n_word_correct += n_correct
        total_loss += loss.item()
        i += 1

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def eval_epoch(model, tokenizer, validation_data, device):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    desc = '  - (Validation) '
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc=desc, leave=False):

            # prepare data
            src_seqs = batch['proteins'] # batch_size x len
            ids = tokenizer.batch_encode_plus(src_seqs, add_special_tokens=True, padding='longest')
            input_ids = torch.tensor(ids['input_ids']).to(device)
            attention_mask = torch.tensor(ids['attention_mask']).to(device)

            trg_seqs, gold = map(lambda x: x.to(device), patch_trg(batch['drugs'])) # batch_size x len      1 x (batch_size x len)

            # forward
            pred = model(input_ids, trg_seqs, attention_mask)

            loss, n_correct, n_word = cal_performance(pred, gold, 0, smoothing=False)

            # note keeping
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.item()

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def train(model, tokenizer, training_data, validation_data, optimizer, scheduler, device, arg):
    ''' Start training '''

    log_train_file = os.path.join(arg['output_dir'], 'train.log')
    log_valid_file = os.path.join(arg['output_dir'], 'valid.log')

    print('[Info] Training performance will be written to file: {} and {}'.format(
        log_train_file, log_valid_file))

    with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
        log_tf.write('epoch,loss,ppl,accuracy\n')
        log_vf.write('epoch,loss,ppl,accuracy\n')

    def print_performances(header, ppl, accu, start_time, lr):
        print('  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, lr: {lr:8.5f}, '\
              'elapse: {elapse:3.3f} min'.format(
                  header=f"({header})", ppl=ppl,
                  accu=100*accu, elapse=(time.time()-start_time)/60, lr=lr))

    valid_losses = []
    for epoch_i in range(arg['epoch']):
        ################## training ###################
        print('[ Epoch', epoch_i, ']')
        start = time.time()
        train_loss, train_accu = train_epoch(model, tokenizer, training_data, optimizer, device, smoothing=arg['label_smoothing'])
        train_ppl = math.exp(min(train_loss, 100))
        scheduler.step()
        lr = optimizer.param_groups[0]['lr'] # Current learning rate
        print_performances('Training', train_ppl, train_accu, start, lr)

        ################# validating #################
        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, tokenizer, validation_data, device)
        valid_ppl = math.exp(min(valid_loss, 100))
        print_performances('Validation', valid_ppl, valid_accu, start, lr)
        valid_losses += [valid_loss]

        checkpoint = {'epoch': epoch_i, 'settings': arg, 'model': model.state_dict()}

        if arg['save_mode'] == 'all':
            model_name = 'model_epoch_{}.chkpt'.format(epoch_i)
            torch.save(checkpoint, os.path.join(arg['output_dir'], model_name))
        elif arg['save_mode'] == 'best':
            model_name = 'model.chkpt'
            if valid_loss <= min(valid_losses):
                torch.save(checkpoint, os.path.join(arg['output_dir'], model_name))
                print('    - [Info] The checkpoint file has been updated.')

        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
            log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                epoch=epoch_i, loss=train_loss,
                ppl=train_ppl, accu=100*train_accu))
            log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                epoch=epoch_i, loss=valid_loss,
                ppl=valid_ppl, accu=100*valid_accu))

def main(arg:dict)->None:

    if arg['seed'] is not None:
        torch.manual_seed(arg['seed'])
        torch.backends.cudnn.benchmark = False
        np.random.seed(arg['seed'])
        random.seed(arg['seed'])

    if not os.path.exists(arg['output_dir']):
        os.makedirs(arg['output_dir'])

    device = torch.device('cuda' if arg['cuda'] else 'cpu')

    ################## loading checkpoint ##################
    checkpoint = torch.load(arg['load_path'], map_location=device)

    new_arg = copy.deepcopy( checkpoint["settings"])
    for key in arg:
        new_arg[key] = arg[key]
    
    arg = new_arg

    ################## loading dataset #####################
    print('..........Loading Vocabulary..........')
    drug_vocab    = vocabulary.Vocabulary(arg['vocab_path_drug'])
    protein_vocab = vocabulary.Vocabulary(arg['vocab_path_protein'])

    print('..........Loading Dataset...........')
    dataset_pkl = pickle.load(open(arg['pair_data_pkl'],'rb'))
    trainset = dataset.ProtransDataset(dataset_pkl['trainset'], protein_vocab, device)
    validset = dataset.ProtransDataset(dataset_pkl['validset'], protein_vocab, device)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=arg['batch_size'], shuffle=True, num_workers= 4, collate_fn = dataset.ProtransDataset.collate_fn)
    validloader = torch.utils.data.DataLoader(validset, batch_size=arg['batch_size'], shuffle=True, num_workers= 4, collate_fn = dataset.ProtransDataset.collate_fn)
    ################# preparing model #####################
    print('..........Preparing Model...........')

    bert = BertModel.from_pretrained("Rostlab/prot_bert_bfd")
    bert = bert.to(device)
    tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False)

    transformer = model.ProtTrans(
        None,
        drug_vocab.get_size(),
        src_pad_idx=0,
        trg_pad_idx=0,
        trg_emb_prj_weight_sharing=arg['proj_share_weight'],
        emb_src_trg_weight_sharing=arg['embs_share_weight'],
        d_k=arg['d_k'],
        d_v=arg['d_v'],
        d_model=arg['d_model'],
        d_word_vec=arg['d_model'],
        d_inner=arg['d_inner_hid'],
        n_layers=arg['n_layers'],
        n_head=arg['n_head'],
        n_position_src=arg['maxlen_protein'],
        n_position_trg=arg['maxlen_drug'],
        dropout=arg['dropout'],
        scale_emb_or_prj=arg['scale_emb_or_prj']).to(device)

    transformer.load_state_dict(checkpoint['model'])

    transformer.bert = bert

    optimizer = optim.Adam(transformer.parameters(), lr = arg['lr'], betas=(0.9, 0.98), eps=1e-09)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    train(transformer, tokenizer, trainloader, validloader, optimizer, scheduler, device, arg)

    ################################################################

if __name__ == '__main__':
    arg = {}
    # 预训练模型路径
    arg['load_path']              = 'save_folder/protrans_pretrain/d_transformer_token_token_b64_l0.0001_e50/model.chkpt'
  
    # 训练后保存的模块
    arg['output_dir']             = 'save_folder/protrans_finetune/protrans_e_bert_d_transformer_token_token_l0.00001_e30/'
    arg['save_mode']              = 'all'   
    
    # train_config
    arg['cuda']                   = True
    arg['label_smoothing']        = False
    arg['seed']                   = 0
    arg['epoch']                  = 30
    arg['batch_size']             = 16   
    arg['step_batch']             = 4
    arg['lr']                     = 1e-5
    main(arg)

