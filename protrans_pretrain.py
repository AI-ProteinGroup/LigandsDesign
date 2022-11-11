'''
This script handles the training process.
'''
from copy import deepcopy
import math
import os
from pickle import FALSE
import random
import time
from tqdm import tqdm
import dill as pickle
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from utils import vocabulary, dataset
import transformer.Models as model
import transformer.Optim  as op
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def cal_performance(pred, gold, trg_pad_idx = 0, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word

def cal_loss(pred, gold, trg_pad_idx = 0, smoothing=False):
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

def train_epoch(model, training_data, optimizer, device, smoothing, arg):
    ''' Epoch operation in training phase'''

    model.train()
    total_loss, n_word_total, n_word_correct = 0, 0, 0 

    desc = '  - (Training)   '
    for batch in tqdm(training_data, mininterval=2, desc=desc, leave=False):
        # prepare data
        src_seqs = batch['seqs'].to(device) 
        trg_seqs = batch['labels'].to(device)

        # forward
        optimizer.zero_grad()
        if arg['encoder']:
            pred = model.pretrain_encoder(src_seqs)
        else:
            pred = model.pretrain_decoder(src_seqs)

        # backward and update parameters
        loss, n_correct, n_word = cal_performance(pred, trg_seqs, 0, smoothing=smoothing) 
        loss.backward()
        optimizer.step()
        
        # note keeping
        n_word_total += n_word
        n_word_correct += n_correct
        total_loss += loss.item()

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def eval_epoch(model, validation_data, device, arg):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    desc = '  - (Validation) '
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc=desc, leave=False):

            # prepare data
            src_seqs = batch['seqs'].to(device) 
            trg_seqs = batch['labels'].to(device)

            # forward
            if arg['encoder']:
                pred = model.pretrain_encoder(src_seqs)
            else:
                pred = model.pretrain_decoder(src_seqs)

            # backward and update parameters
            loss, n_correct, n_word = cal_performance(pred, trg_seqs) 

            # note keeping
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.item()

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def pretrain(model, training_data, validation_data, optimizer, scheduler, device, arg):
    ''' Start training '''
    log_train_file = os.path.join(arg['output_dir'], 'train.log')
    log_valid_file = os.path.join(arg['output_dir'], 'valid.log')

    print('[Info] Training performance will be written to file: {} and {}'.format(
        log_train_file, log_valid_file))

    with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
        log_tf.write('epoch,loss,accuracy\n')
        log_vf.write('epoch,loss,accuracy\n')

    def print_performances(header, accu, start_time, lr):
        print('  - {header:12}  accuracy: {accu:3.3f} %, lr: {lr:8.5f}, '\
              'elapse: {elapse:3.3f} min'.format(
                  header=f"({header})",
                  accu=100*accu, elapse=(time.time()-start_time)/60, lr=lr))

    valid_losses = []
    for epoch_i in range(arg['epoch']):
        ################## training ###################
        print('[ Epoch', epoch_i, ']')
        start = time.time()
        train_loss, train_accu = train_epoch(model, training_data, optimizer, device, arg['label_smoothing'], arg)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr'] # Current learning rate
        print_performances('Training', train_accu, start, lr)

        ################# validating #################
        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, device, arg)
        print_performances('Validation', valid_accu, start, lr)
        valid_losses += [valid_loss]

        checkpoint = {'epoch': epoch_i, 'settings': arg, 'model': model.transformer.state_dict()}

        if arg['save_mode'] == 'all':
            model_name = 'model_epoch_{}.chkpt'.format(epoch_i)
            torch.save(checkpoint, os.path.join(arg['output_dir'], model_name))
        elif arg['save_mode'] == 'best':
            model_name = 'model.chkpt'
            if valid_loss <= min(valid_losses):
                torch.save(checkpoint, os.path.join(arg['output_dir'], model_name))
                print('    - [Info] The checkpoint file has been updated.')

        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
            log_tf.write('{epoch},{loss: 8.5f},{accu:3.3f}\n'.format(
                epoch=epoch_i, loss=train_loss,
                 accu=100*train_accu))
            log_vf.write('{epoch},{loss: 8.5f},{accu:3.3f}\n'.format(
                epoch=epoch_i, loss=valid_loss,
                 accu=100*valid_accu))


def main(arg:dict)->None:

    if arg['seed'] is not None:
        torch.manual_seed(arg['seed'])
        torch.backends.cudnn.benchmark = False
        np.random.seed(arg['seed'])
        random.seed(arg['seed'])

    if not os.path.exists(arg['output_dir']):
        os.makedirs(arg['output_dir'])

    device = torch.device('cuda' if arg['cuda'] else 'cpu')

    ################## loading dataset #####################
    print('..........Loading Vocabulary..........')
    drug_vocab    = vocabulary.Vocabulary(arg['vocab_path_drug'])
    protein_vocab = vocabulary.Vocabulary(arg['vocab_path_protein'])

    print('..........Loading Dataset...........')
    dataset_pkl = pickle.load(open(arg['pair_data_pkl'],'rb'))
    trainset = dataset.PairDataset(dataset_pkl['trainset'])
    validset = dataset.PairDataset(dataset_pkl['validset'])
    testset  = dataset.PairDataset(dataset_pkl['testset'])

    protein_pkl = pickle.load(open(arg['mono_protein_pkl'], 'rb'))
    protein_trainset = dataset.MonoDataset(protein_pkl['trainset'], protein_vocab)
    protein_validset = dataset.MonoDataset(protein_pkl['validset'], protein_vocab)
    drug_pkl = pickle.load(open(arg['mono_drug_pkl'], 'rb'))
    drug_trainset = dataset.MonoDataset(drug_pkl['trainset'], drug_vocab)
    drug_validset = dataset.MonoDataset(drug_pkl['validset'], drug_vocab)
    arg['maxlen_drug']    = max(trainset.get_max_len()['drug'], validset.get_max_len()['drug'], testset.get_max_len()['drug'], drug_trainset.get_max_len(), drug_validset.get_max_len())
    arg['maxlen_protein'] = max(trainset.get_max_len()['protein'], validset.get_max_len()['protein'], testset.get_max_len()['protein'], protein_trainset.get_max_len(), protein_validset.get_max_len())
    print('maxlen_drug:{} \nmaxlen_protein:{}'.format(arg['maxlen_drug'], arg['maxlen_protein']))

    if arg['encoder']:
        trainloader = torch.utils.data.DataLoader(protein_trainset, batch_size=arg['batch_size'], shuffle=True, num_workers= 4, collate_fn = dataset.MonoDataset.collate_fn)
        validloader = torch.utils.data.DataLoader(protein_validset, batch_size=arg['batch_size'], shuffle=True, num_workers= 4, collate_fn = dataset.MonoDataset.collate_fn)
    else:
        trainloader = torch.utils.data.DataLoader(drug_trainset, batch_size=arg['batch_size'], shuffle=True, num_workers= 4, collate_fn = dataset.MonoDataset.collate_fn)
        validloader = torch.utils.data.DataLoader(drug_validset, batch_size=arg['batch_size'], shuffle=True, num_workers= 4, collate_fn = dataset.MonoDataset.collate_fn)

    ################# preparing model #####################
    print('..........Preparing Model...........')
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

    bert = model.Bert(transformer = transformer, 
                      d_model=arg['d_model'],
                      d_inner=arg['d_inner_hid'],
                      n_src_vocab = protein_vocab.get_size(),
                      n_trg_vocab = drug_vocab.get_size(),
                      dropout = arg['dropout']
                    ).to(device)

    if arg['load'] is True:
        checkpoint = torch.load(arg['load_path'], map_location=device)
        bert.load_state_dict(checkpoint['model'])

    optimizer = optim.Adam(transformer.parameters(), lr = arg['lr'], betas=(0.9, 0.98), eps=1e-09)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)

    pretrain(bert, trainloader, validloader, optimizer, scheduler, device, arg)


    ################################################################

if __name__ == '__main__':
    arg = {}
    # 是否要加载
    arg['load']                   = False
    arg['load_path']              = 'save_folder/pretrain/e_transformer_token_token_b64_l0.0001_e50/model.chkpt'
    
    # 预训练哪个模块
    arg['encoder']                = False

    # 数据集
    arg['pair_data_pkl']          = 'data/tokenized/pair/dataset_token_token.pkl' 
    arg['mono_protein_pkl']       = 'data/tokenized/mono/protein_token.pkl'
    arg['mono_drug_pkl']          = 'data/tokenized/mono/drug_token.pkl'

    # 使用的词库
    arg['vocab_path_drug']        = 'data/vocabulary/vocabulary_smiles.txt'
    arg['vocab_path_protein']     = 'data/vocabulary/vocabulary_proteins.txt'
    
    # 训练后保存的模块
    arg['output_dir']             = 'save_folder/protrans/d_transformer_token_token_b64_l0.0001_e50/'
    arg['save_mode']              = 'best'   
    
    # train_config
    arg['cuda']                   = True
    arg['label_smoothing']        = False
    arg['seed']                   = 0
    arg['epoch']                  = 50
    arg['batch_size']             = 64    
    arg['lr']                     = 1e-4

    # model_config
    arg['d_model']                = 1024
    arg['d_inner_hid']            = 2048
    arg['n_layers']               = 6       
    arg['n_head']                 = 8      
    arg['d_k']                    = 64
    arg['d_v']                    = 64
    arg['dropout']                = 0.1
    arg['embs_share_weight']      = False
    arg['proj_share_weight']      = False
    arg['scale_emb_or_prj']       = 'prj'
    main(arg)
