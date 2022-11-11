import os
import re
import random
import numpy as np
import pandas as pd
import torch
import dill as pickle
from tqdm import tqdm

import transformer.Models as model
from transformer.Translator import ProtransTranslator
from utils import vocabulary
from transformers import BertModel, BertTokenizer

def amino_acid_wise_tokenizer(sequence):
    """
    Tokenize a protein sequence at amino-acid-wise

    sequence: A amino-acid sequence
    """

    pattern =  "(A|R|N|D|C|Q|E|G|H|I|L|K|M|F|P|S|T|W|Y|V)" # 登记所有要匹配的字符,这里为20种人体内氨基酸。部分数据库包含X,代表未知氨基酸

    regex = re.compile(pattern) # 编译成正则表达式模板
    tokens = [token for token in regex.findall(sequence)] # findall会返回一个list,每个元素都是一个氨基酸符号

    return tokens

def main(arg:dict):

    if arg['seed'] is not None:
        torch.manual_seed(arg['seed'])
        torch.backends.cudnn.benchmark = False
        np.random.seed(arg['seed'])
        random.seed(arg['seed'])

    if arg['generate'] is True:   # 是否要生成，还是直接读生成的文件
        if not os.path.exists(arg['output_dir']):
            os.makedirs(arg['output_dir'])

        print("..........Loading ProtBert..........")
        device = torch.device('cuda:0' if arg['cuda'] else 'cpu')
        bert = BertModel.from_pretrained("Rostlab/prot_bert_bfd")
        bert = bert.to(device)
        tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False )

        checkpoint = torch.load(arg['model'], map_location=device)
        model_opt = checkpoint['settings']

        drug_vocab    = vocabulary.Vocabulary(model_opt['vocab_path_drug'])
        protein_vocab = vocabulary.Vocabulary(model_opt['vocab_path_protein'])

        ###################### loading model  ########################  
        transformer = model.ProtTrans(
            bert,
            drug_vocab.get_size(),
            src_pad_idx=0,
            trg_pad_idx=0,
            trg_emb_prj_weight_sharing=model_opt['proj_share_weight'],
            emb_src_trg_weight_sharing=model_opt['embs_share_weight'],
            d_k=model_opt['d_k'],
            d_v=model_opt['d_v'],
            d_model=model_opt['d_model'],
            d_word_vec=model_opt['d_model'],
            d_inner=model_opt['d_inner_hid'],
            n_layers=model_opt['n_layers'],
            n_head=model_opt['n_head'],
            n_position_src=model_opt['maxlen_protein'],
            n_position_trg=model_opt['maxlen_drug'],
            dropout=model_opt['dropout'],
            scale_emb_or_prj=model_opt['scale_emb_or_prj']).to(device)

        if arg["load"] is True:
            transformer.load_state_dict(checkpoint['model'])
            print('[Info] Trained model state loaded.')

        translator = ProtransTranslator(
            model =  transformer,
            beam_size   = 5,
            max_seq_len = arg['max_seq_len'],
            trg_pad_idx = 0,
            trg_bos_idx = 1,
            trg_eos_idx = 2).to(device)
    
        protein_list = []
        drug_list = []
        proteins = [arg['protein']]
        for protein in list(proteins):
            proteins = ' '.join(amino_acid_wise_tokenizer(protein))

            ids = tokenizer.batch_encode_plus([proteins], add_special_tokens=True, padding='longest')
            input_ids = torch.tensor(ids['input_ids']).to(device)
            attention_mask = torch.tensor(ids['attention_mask']).to(device)

            for i in tqdm(range(arg['num'])):
                pred_seq = translator.translate_sentence(input_ids, attention_mask, arg["random"])
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

    

if __name__ == "__main__":
    arg = {}
    arg['seed']          = 0
    arg['cuda']          = True

    # dataset
    arg['protein']      = 'SEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLKECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTLSEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLVAASQAA'
    
    # 这次是否需要生成
    arg['generate']      = True

    # 生成的话是否要加载模型
    arg['load']          = True
    arg['model']         = 'save_folder/protrans_finetune/protrans_e_bert_d_transformer_token_token_l0.00001_e30/model_epoch_29.chkpt'
    arg['max_seq_len']   = 150
    arg['num']           = 10000
    arg['random']        = 1

    # 生成分子的文件
    arg['output_dir']    = 'result/translate_on_albumin/'

    main(arg)

