################### two base tokenizer ##########################
# 1. atom_wise_tokenizer(smi)                 for  smiles sequence
# 2. amino_acid_wise_tokenizer(sequence)      for  protein sequence
import re
from typing import List  


def atom_wise_tokenizer(smi):
    """
    Tokenize a SMILES molecule at atom-level:

    smi: A smi string
    """
    ret = []
    for c in smi:
        ret.append(c)
    return ret

def amino_acid_wise_tokenizer(sequence):
    """
    Tokenize a protein sequence at amino-acid-wise

    sequence: A amino-acid sequence
    """

    pattern =  "(A|R|N|D|C|Q|E|G|H|I|L|K|M|F|P|S|T|W|Y|V)" # 登记所有要匹配的字符,这里为20种人体内氨基酸。部分数据库包含X,代表未知氨基酸

    regex = re.compile(pattern) # 编译成正则表达式模板
    tokens = [token for token in regex.findall(sequence)] # findall会返回一个list,每个元素都是一个氨基酸符号

    return tokens

################## two tokenizer ############################
import pandas as pd
import vocabulary
import pickle

class Tokenizer(object):
    def __init__(
                self,
                train_path,
                valid_path,
                test_path,
                output_path,
                vocabulry_smiles_path,
                vocabulry_proteins_path) -> None:

        # 读取csv
        train_df = pd.read_csv(train_path)
        valid_df = pd.read_csv(valid_path)
        test_df  = pd.read_csv(test_path)

        # 纯字符串
        train_drugs    = list(train_df["Ligand SMILES"])
        train_proteins = list(train_df["BindingDB Target Chain  Sequence"])
        valid_drugs    = list(valid_df["Ligand SMILES"])
        valid_proteins = list(valid_df["BindingDB Target Chain  Sequence"])
        test_drugs     = list(test_df["Ligand SMILES"])
        test_proteins  = list(test_df["BindingDB Target Chain  Sequence"])

        train_drugs, train_proteins = self.token_encode(train_drugs, train_proteins)
        valid_drugs, valid_proteins = self.token_encode(valid_drugs, valid_proteins) 
        test_drugs, test_proteins = self.token_encode(test_drugs, test_proteins)

        # 根据词库将word转成编号
        vocab_drugs    = vocabulary.Vocabulary(vocabulry_smiles_path)
        vocab_proteins = vocabulary.Vocabulary(vocabulry_proteins_path)

        train_drugs, train_proteins = self.to_index(vocab_drugs, train_drugs, vocab_proteins, train_proteins)
        valid_drugs, valid_proteins = self.to_index(vocab_drugs, valid_drugs, vocab_proteins, valid_proteins)
        test_drugs,  test_proteins  = self.to_index(vocab_drugs, test_drugs,  vocab_proteins, test_proteins)

        data = {
            'trainset' : {
                'drugs' : train_drugs,
                'proteins' : train_proteins
            },
            'validset' : {
                'drugs' : valid_drugs,
                'proteins' : valid_proteins
            },
            'testset'  : {
                'drugs' : test_drugs,
                'proteins' : test_proteins
            }
        }

        pickle.dump(data, open(output_path, 'wb'))

    def to_index(self, vocab_drug:vocabulary.Vocabulary, drug_seqs:list, vocab_protein:vocabulary.Vocabulary, protein_seqs:list)->list:
        '''
        change every word into index
        example:  a vocabulary ['a', 'b' ,'c'] than a sequence ['b', 'a', 'c'] will become [1, 0, 2]

        input:
            vocab: a Vocabulary object
            seqs:  a list of sequences.   Each sequence is a list of words
        return:
            indexs: list of list of int
        '''
        drugs = []
        proteins = []
        for (drug_seq, protein_seq) in zip(drug_seqs, protein_seqs):
            drug_indexs = []
            protein_indexs = []
            try:
                for word in drug_seq:
                    drug_indexs.append(vocab_drug.to_index(word))

                for word in protein_seq:
                    protein_indexs.append(vocab_protein.to_index(word))
            except:
                continue
                
            drugs.append(drug_indexs)
            proteins.append(protein_indexs)
        return (drugs, proteins)
                
    def token_encode(self, drugs:list, proteins:list):
        '''
        input:
            drugs:    list of smiles strings
            proteins: list of proteins string

        return:
            ret_drugs :   list of list of words
            ret_proteins: list of list of words
        '''
        ret_drugs = []
        ret_proteins = []

        # 两边加上eos和bos
        for smi in drugs:
            temp = []
            temp.append('bos')
            temp.extend(atom_wise_tokenizer(smi))
            temp.append('eos')
            ret_drugs.append(temp)

        for protein in proteins:
            temp = []
            temp.append('bos')
            temp.extend(amino_acid_wise_tokenizer(protein))
            temp.append('eos')
            ret_proteins.append(temp)

        return tuple([ret_drugs, ret_proteins])

class MonoTokenizer(object):
    def __init__(
                self,
                input_path,
                output_path,
                is_excel,    # 蛋白是excel文件，分子是txt文件
                is_drug,
                vocabulry_path) -> None:

        # 读取excel或txt
        seqs = []
        if is_excel:       
            df = pd.read_excel(input_path)
            seqs = list(df["Sequence"])
        else:
            with open(input_path, 'r') as f:
                for line in f.readlines():
                    if is_drug: 
                        if len(line) < 150:
                            seqs.append(line.strip('\n'))
                    else:
                        if len(line) < 2050:
                            seqs.append(line.strip('\n'))
                    
        (train_seqs, valid_seqs) = self.token_encode(seqs, is_drug)

        # 根据词库将word转成编号
        vocab  = vocabulary.Vocabulary(vocabulry_path)

        train_seqs = self.to_index(vocab, train_seqs)
        valid_seqs = self.to_index(vocab, valid_seqs)

        print(len(train_seqs) + len(valid_seqs))

        data = {
            'trainset' : train_seqs,
            'validset' : valid_seqs
        }

        pickle.dump(data, open(output_path, 'wb'))

    def to_index(self, vocab:vocabulary.Vocabulary, seqs:list)->list:
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
                
    def token_encode(self, seqs:list, is_drug:bool):
        ret = []

        # 两边加上eos和bos
        for seq in seqs:
            temp = []
            temp.append('bos')
            if is_drug:
                temp.extend(atom_wise_tokenizer(seq))
            else:
                temp.extend(amino_acid_wise_tokenizer(seq))
            temp.append('eos')
            ret.append(temp)
        return ret[:int(len(ret)*0.8)], ret[int(len(ret)*0.8):]

        
if __name__ == "__main__":
    '''
    # use for test api

    SPE = SPE_Tokenizer("../data/bpe_smiles.txt")
    result = SPE.tokenize("N=C(N)c1ccc2c(c1)c(CC(=O)O)cn2CC(=O)N1CCC(Cc2ccccc2)CC1")
    print(result)

    PPE = PPE_Tokenizer("../data/bpe_proteins.txt")
    result = PPE.tokenize("".join("M R L P G A M P A L A L K G E L L L L S L L L L L E P Q I S Q G L V V T P P G P E L V L N V S S T F V L T C S G S A P V V W E R M S Q E P P Q E M A K A Q D G T F S S V L T L T N L T G L D T G E Y F C T H N D S R G L E T D E R K R L Y I F V P D P T V G F L P N D A E E L F I F L T E I T E I T I P C R V T D P Q L V V T L H E K K G D V A L P V P Y D H Q R G F S G I F E D R S Y I C K T T I G D R E V D S D A Y Y V Y R L Q V S S I N V S V N A V Q T V V R Q G E N I T L M C I V I G N E V V N F E W T Y P R K E S G R L V E P V T D F L L D M P Y H I R S I L H I P S A E L E D S G T Y T C N V T E S V N D H Q D E K A I N I T V V E S G Y V R L L G E V G T L Q F A E L H R S R T L Q V V F E A Y P P P T V L W F K D N R T L G D S S A G E I A L S T R N V S E T R Y V S E L T L V R V K V A E A G H Y T M R A F H E D A E V Q L S F Q L Q I N V P V R V L E L S E S H P D S G E Q T V R C R G R G M P Q P N I I W S A C R D L K R C P R E L P P T L L G N S S E E E S Q L E T N V T Y W E E E Q E F E V V S T L R L Q H V D R P L S V R C T L R N A V G Q D T Q E V I V V P H S L P F K V V V I S A I L A L V V L T I I S L I I L I M L W Q K K P R Y E I R W K V I E S V S S D G H E Y I Y V D P M Q L P Y D S T W E L P R D Q L V L G R T L G S G A F G Q V V E A T A H G L S H S Q A T M K V A V K M L K S T A R S S E K Q A L M S E L K I M S H L G P H L N V V N L L G A C T K G G P I Y I I T E Y C R Y G D L V D Y L H R N K H T F L Q H H S D K R R P P S A E L Y S N A L P V G L P L P S H V S L T G E S D G G Y M D M S K D E S V D Y V P M L D M K G D V K Y A D I E S S N Y M A P Y D N Y V P S A P E R T C R A T L I N E S P V L S Y M D L V G F S Y Q V A N G M E F L A S K N C V H R D L A A R N V L I C E G K L V K I C D F G L A R D I M R D S N Y I S K G S T F L P L K W M A P E S I F N S L Y T T L S D V W S F G I L L W E I F T L G G T P Y P E L P M N E Q F Y N A I K R G Y R M A Q P A H A S D E I Y E I M Q K C W E E K F E I R P P F S Q L V L L L E R L L G E G Y K K K Y Q Q V D E E F L R S D H P A I L R S Q A R L P G F H G L R S P L D T S S V L Y T A V Q P N E G D N D Y I I P L P D P K P E V A D E G P L E G S P S L A S S T L N E V N T S S T I S C D S P L E P Q D E P E P E P Q L E L Q V E P E P E L E Q L P D S G C P A P R A E A E D S F L".split(" ")))
    print(result)
    '''

    # 将用于seq2seq训练的pair数据进行处理
    tokenizer = Tokenizer(  '../data/dataset_train.csv', 
                            '../data/dataset_valid.csv', 
                            '../data/dataset_test.csv', 
                            '../data/tokenized/pair/dataset_token_token.pkl',
                            '../data/vocabulary/vocabulary_smiles.txt',
                            '../data/vocabulary/vocabulary_proteins.txt')

    # 将用于bert预训练的分子序列进行处理
    tokenizer = MonoTokenizer('../data/chembl_28_pxc_5.txt',
                              '../data/tokenized/mono/drug_token.pkl',
                              False,
                              True,
                              '../data/vocabulary/vocabulary_smiles.txt')

    # 将用于bert预训练的蛋白序列进行处理
    tokenizer = MonoTokenizer('../data/uniprot_human_all.xlsx',
                              '../data/tokenized/mono/protein_token.pkl',
                              True,
                              False,
                              '../data/vocabulary/vocabulary_proteins.txt')

    

