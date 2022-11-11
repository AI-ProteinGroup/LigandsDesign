import os
import re
import random
import numpy as np
import pandas as pd
import torch
import dill as pickle
from tqdm import tqdm

from transformer.Models import Transformer
from transformer.Translator import Translator
from utils import vocabulary


def amino_acid_wise_tokenizer(sequence):
    """
    Tokenize a protein sequence at amino-acid-wise

    sequence: A amino-acid sequence
    """

    pattern =  "(A|R|N|D|C|Q|E|G|H|I|L|K|M|F|P|S|T|W|Y|V)" # 登记所有要匹配的字符,这里为20种人体内氨基酸。部分数据库包含X,代表未知氨基酸

    regex = re.compile(pattern) # 编译成正则表达式模板
    tokens = [token for token in regex.findall(sequence)] # findall会返回一个list,每个元素都是一个氨基酸符号

    return tokens

def token_encode(seqs:list):
        ret = []

        # 两边加上eos和bos
        for seq in seqs:
            temp = []
            temp.append('bos')
            temp.extend(amino_acid_wise_tokenizer(seq))
            temp.append('eos')
            ret.append(temp)
        return ret

def to_index(vocab:vocabulary.Vocabulary, seqs:list)->list:
        '''
        change every word into index
        example:  a vocabulary ['a', 'b' ,'c'] then a sequence ['b', 'a', 'c'] will become [1, 0, 2]

        input:
            vocab: a Vocabulary object
            seqs:  a list of sequences.   Each sequence is a list of words
        return:
            indexs: list of list of int
        '''
        ret = []
        for seq in seqs:
            indexs = []
            try:
                for word in seq:
                    indexs.append(vocab.to_index(word))
            except:
                continue
                
            ret.append(indexs)
        return ret

def main(arg:dict):

    if arg['seed'] is not None:
        torch.manual_seed(arg['seed'])
        torch.backends.cudnn.benchmark = False
        np.random.seed(arg['seed'])
        random.seed(arg['seed'])

    ###################### loading dataset #######################
    print('..........Loading Dataset...........')

    df = pd.read_csv(arg['data_dir'])
    data = {}
    ids = {}
    proteins = set()
    for index, row in df.iterrows():
        protein = row['BindingDB Target Chain  Sequence']
        drug = row['Ligand SMILES']
        protein_id = row['UniProt (SwissProt) Primary ID of Target Chain']

        proteins.add(protein)
        if protein not in data:
            data[protein] = []
            ids[protein] = protein_id
        
        data[protein].append(drug)

    if arg['generate'] is True:   # 是否要生成，还是直接读生成的文件
        if not os.path.exists(arg['output_dir']):
            os.makedirs(arg['output_dir'])

        device = torch.device('cuda:0' if arg['cuda'] else 'cpu')

        checkpoint = torch.load(arg['model'], map_location=device)
        model_opt = checkpoint['settings']

        drug_vocab    = vocabulary.Vocabulary(model_opt['vocab_path_drug'])
        protein_vocab = vocabulary.Vocabulary(model_opt['vocab_path_protein'])

        ###################### loading model  ########################  
        model = Transformer(
            n_src_vocab= protein_vocab.get_size(),
            n_trg_vocab= drug_vocab.get_size(),
            src_pad_idx= 0,
            trg_pad_idx= 0,
            trg_emb_prj_weight_sharing=model_opt['proj_share_weight'],
            emb_src_trg_weight_sharing=model_opt['embs_share_weight'],
            d_k=model_opt['d_k'],
            d_v=model_opt['d_v'],
            d_model=model_opt['d_model'],
            d_word_vec=model_opt['d_model'],
            d_inner=model_opt['d_inner_hid'],
            n_layers=model_opt['n_layers'],
            n_head=model_opt['n_head'],
            dropout=model_opt['dropout'],
            n_position_src=model_opt['maxlen_protein'],
            n_position_trg=model_opt['maxlen_drug'],
            scale_emb_or_prj=model_opt['scale_emb_or_prj']).to(device)

        if arg["load"] is True:
            model.load_state_dict(checkpoint['model'])
            print('[Info] Trained model state loaded.')

        translator = Translator(
            model =  model,
            beam_size   = 5,
            max_seq_len = arg['max_seq_len'],
            src_pad_idx = 0,
            trg_pad_idx = 0,
            trg_bos_idx = 1,
            trg_eos_idx = 2).to(device)
    
        protein_list = []
        drug_list = []
        for protein in list(proteins):
            proteins = token_encode([protein])
            proteins = to_index(protein_vocab, proteins)
            protein_tensor = torch.tensor(proteins).view(1, -1)

            for i in tqdm(range(arg['num'])):
                src_seqs = protein_tensor.to(device)   # batch_size = 1 实际是1xlen
                pred_seq = translator.translate_sentence(src_seqs, arg["random"])
                pred_line = ''.join(drug_vocab.to_word(idx) for idx in pred_seq)
                pred_line = pred_line.replace('pad', '').replace('bos', '').replace('eos','') # 去掉bos
                protein_list.append(protein)
                drug_list.append(pred_line)

        dataframe = pd.DataFrame({'protein':protein_list, 'drug':drug_list})
        dataframe.to_csv(arg['output_dir'] + 'generated.csv')
        print('[Info] Finished.')

    gdf = pd.read_csv(arg['output_dir'] + "generated.csv")
    generated_data = {}
    for index, row in gdf.iterrows():
        protein = row['protein']
        drug = row['drug']

        if protein not in generated_data:
            generated_data[protein] = []
        
        generated_data[protein].append(drug)
    
    import moses
    result = {}
    for key in generated_data:
        model_metrics = moses.get_all_metrics(gen=generated_data[key],test=data[key],test_scaffolds=data[key], train = [])
        result[ids[key]] = model_metrics
    
    print(result)

    

if __name__ == "__main__":
    arg = {}
    arg['seed']          = 0
    arg['cuda']          = True

    # dataset
    arg['data_dir']      = 'data/proteins_and_ligands_experiment.csv'
    
    # 这次是否需要生成
    arg['generate']      = False

    # 生成的话是否要加载模型
    arg['load']          = False
    arg['model']         = 'save_folder/bert_finetune/e_d_transformer_token_token_l0.00001_e30/model_epoch_4.chkpt'
    arg['max_seq_len']   = 150
    arg['num']           = 10000
    arg['random']        = 100

    # 生成分子的文件
    arg['output_dir']    = 'result/translate_on_experimentset/transformer_token_token_l0.00001_e0_random100/'

    main(arg)

