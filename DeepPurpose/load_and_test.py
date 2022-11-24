from copy import deepcopy
from pickle import STRING
import random
import os
import warnings
import pandas as pd
from DeepPurpose import DTI as models 
from DeepPurpose import utils, dataset
from rdkit import Chem
from matplotlib import pyplot as plt
os.chdir('../')
warnings.filterwarnings("ignore")

def valid_score(smiles):
    """
    score a smiles , if  it is valid, score = 1 ; else score = 0
    Parameters
    ----------
        smiles: str
            SMILES strings 
    Returns
    -------
        score: int 0 or 1
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    else :
        return 1

def main(arg):
    if arg["test_what?"] == "deeppurpose":
        # 1. read data
        print("##########################################")
        print("Loading data...")
        test_file = pd.read_csv(arg["test_path"])
        X_drugs, X_targets = list(test_file["Ligand SMILES"]), list(test_file["BindingDB Target Chain  Sequence"])
        assert len(X_drugs) == len(X_targets)
        print("There are {} pairs".format(len(X_drugs)))

        # 2. load model
        print("##########################################")
        print("loading model...")
        if arg["model_is_my"] is True:
            net = models.model_pretrained(path_dir = arg["my_model_path"])
        else:
            net = models.model_pretrained(model = arg["model_name"])
        print("config:")
        print(net.config)

        # 3.eval
        print("##########################################")
        print("evaluate dataset...")
        drug_encoding   = arg["drug_encoder"]
        target_encoding = arg["target_encoder"]
        y = [1]*len(X_drugs)
        size = len(y)
        act_5  = 0 
        act_7  = 0

        batch_size = 64
        for index in range(int(size/batch_size)+1):
            if (index+1)*batch_size<size:
                X_pred = utils.data_process(X_drugs[index*batch_size:(index+1)*batch_size], X_targets[index*batch_size:(index+1)*batch_size], y[index*batch_size:(index+1)*batch_size], 
                                    drug_encoding, target_encoding, 
                                    split_method='no_split')
            else:
                X_pred = utils.data_process(X_drugs[index*batch_size:], X_targets[index*batch_size:], y[index*batch_size:], 
                                    drug_encoding, target_encoding, 
                                    split_method='no_split')
            y_pred = net.predict(X_pred)
            for n in y_pred:
                if n >= 7:
                    act_7 += 1
                if n >= 5:
                    act_5 += 1
        print("active 5 ratio:{}".format(act_5/size))
        print("active 7 ratio:{}".format(act_7/size))
        
    # run on env_lfq_DeepPurpose  baode01 or 404 pc
    ##########################################  CNNRNN_CNNRNN_BindingDB_IC_EC_KI_KD  50EPOCH  ##############################
    # pxc 7 pair valid_dataset      >= 5 ratio:   0.9985071198897566
    # pxc 7 pair valid_dataset      >= 7 ratio:   0.7302097687949778
    # pxc 7 pair test_dataset       >= 5 ratio:   0.9945747566618797
    # pxc 7 pair test_dataset       >= 7 ratio:   0.6485958193713101

    # pxc 5 pair valid_dataset      >= 5 ratio:   0.9484410902094114
    # pxc 5 pair valid_dataset      >= 7 ratio:   0.3664979170947675
    # pxc 5 pair test_dataset       >= 5 ratio:   0.9382883720930233
    # pxc 5 pair test_dataset       >= 7 ratio:   0.3299906976744186
    ########################################################################################################################
    

    if arg["test_what?"] == "transformer":
        # 1. read data
        print("##########################################")
        print("Loading data...")
        test_file = pd.read_csv(arg["test_path_t"] + "generated.csv")

        X_drugs, X_targets = list(test_file["drug"]), list(test_file["protein"])
        
        assert len(X_drugs) == len(X_targets)
        print("There are {} pairs".format(len(X_drugs)))

        # 2. load model
        print("##########################################")
        print("loading model...")
        if arg["model_is_my"] is True:
            net = models.model_pretrained(path_dir = arg["my_model_path"])
        else:
            net = models.model_pretrained(model = arg["model_name"])
        print("config:")
        print(net.config)


        # 3.eval
        print("##########################################")
        print("evaluate dataset...")
        valid_count = 0
        for drug in deepcopy(X_drugs):
            valid_count += valid_score(drug)
        print("validity:{}".format(valid_count/len(X_drugs))) 

        drug_encoding   = arg["drug_encoder"]
        target_encoding = arg["target_encoder"]
        y = [1]*len(X_drugs)
        size = len(y)
        act_5  = 0 
        act_7  = 0
        batch_size = 64
        output_y = []
        input_drug = []
        input_target = []
        input_y = []
        for index in range(int(size/batch_size)+1): 
            if (index+1)*batch_size<size:
                input_drug =     X_drugs[index*batch_size:(index+1)*batch_size]
                input_target = X_targets[index*batch_size:(index+1)*batch_size]
                input_y =              y[index*batch_size:(index+1)*batch_size]
            else:
                input_drug =     X_drugs[index*batch_size:]
                input_target = X_targets[index*batch_size:]
                input_y      =         y[index*batch_size:]

            X_pred = utils.data_process(input_drug, input_target, input_y, 
                                        drug_encoding, target_encoding, 
                                        split_method='no_split')             
            y_pred = net.predict(X_pred)
            output_y.extend(y_pred)
            for n in y_pred:
                if n >= 7:
                    act_7 += 1
                if n >= 5:
                    act_5 += 1
        print("active 5 ratio:{}".format(act_5/size))
        print("active 7 ratio:{}".format(act_7/size))

        with open(arg['test_path_t'] + 'deeppurpose.txt', 'w') as f:
            for l in output_y:
                f.write(str(l) + '\n')

        dataframe = pd.DataFrame({'protein':X_targets, 'drug':X_drugs, 'predicted_activity':output_y})
        dataframe = dataframe.sort_values(by = 'predicted_activity', ascending = False)
        dataframe.to_csv(arg['test_path_t'] + 'deeppurpose.csv')

    if arg["test_what?"] == "random":
         # 1. read data
        print("##########################################")
        print("Loading data...")
        smiles = []
        with open(arg["test_path_r"],'r') as f:
            for line in f.readlines():
                smiles.append(line.strip('\n'))
        random.shuffle(smiles)
            
        test_file = pd.read_csv(arg["test_path_t"] + "generated.csv")

        X_drugs, X_targets = list(test_file["drug"]), list(test_file["protein"])
        X_drugs  = smiles[:len(X_drugs)]
        assert len(X_drugs) == len(X_targets)
        print("There are {} pairs".format(len(X_drugs)))

        # 2. load model
        print("##########################################")
        print("loading model...")
        if arg["model_is_my"] is True:
            net = models.model_pretrained(path_dir = arg["my_model_path"])
        else:
            net = models.model_pretrained(model = arg["model_name"])
        print("config:")
        print(net.config)


        # 3.eval
        print("##########################################")
        print("evaluate dataset...")
        valid_count = 0
        for drug in deepcopy(X_drugs):
            valid_count += valid_score(drug)
        print("validity:{}".format(valid_count/len(X_drugs))) 

        drug_encoding   = arg["drug_encoder"]
        target_encoding = arg["target_encoder"]
        y = [1]*len(X_drugs)
        size = len(y)
        act_5  = 0 
        act_7  = 0
        batch_size = 64
        output_y = []
        input_drug = []
        input_target = []
        input_y = []
        for index in range(int(size/batch_size)+1): 
            if (index+1)*batch_size<size:
                input_drug =     X_drugs[index*batch_size:(index+1)*batch_size]
                input_target = X_targets[index*batch_size:(index+1)*batch_size]
                input_y =              y[index*batch_size:(index+1)*batch_size]
            else:
                input_drug =     X_drugs[index*batch_size:]
                input_target = X_targets[index*batch_size:]
                input_y      =         y[index*batch_size:]

            X_pred = utils.data_process(input_drug, input_target, input_y, 
                                        drug_encoding, target_encoding, 
                                        split_method='no_split')             
            y_pred = net.predict(X_pred)
            output_y.extend(y_pred)
            for n in y_pred:
                if n >= 7:
                    act_7 += 1
                if n >= 5:
                    act_5 += 1
        print("active 5 ratio:{}".format(act_5/size))
        print("active 7 ratio:{}".format(act_7/size))

        with open(arg['test_r_output']  + 'deeppurpose.txt', 'w') as f:
            for l in output_y:
                f.write(str(l) + '\n')

        dataframe = pd.DataFrame({'protein':X_targets, 'drug':X_drugs, 'predicted_activity':output_y})
        dataframe = dataframe.sort_values(by = 'predicted_activity', ascending = False)
        dataframe.to_csv(arg['test_r_output']  + 'deeppurpose.csv')

if __name__ == "__main__":
    arg = {}
    # 编码方式，只有固定的几种，去deeppurpose的github看
    arg["drug_encoder"]    = "CNN_RNN"                     
    arg["target_encoder"]  = "CNN_RNN"

    # 加载模型
    # 如果为true，会加载arg['my_model_path']指定的模型。这是我自己训练的
    # 如果为false，会加载arg['model_name']指定的模型。  这是deeppurpose作者提前训练好的，并发表在网上。 
    arg["model_is_my"]     = True                             
    arg["model_name"]      = "CNN_CNN_BindingDB_IC50"      
    arg["my_model_path"]   =  "./saved_model/50"  

    # 选择用deeppurpose进行验证
    # "deeppurpose" -> 测试deeppurpose模型的性能
    # "transformer" -> 预测模型生成的分子和蛋白结合的活性
    # "random"      -> 预测随机挑选的分子和蛋白结合的活性
    arg["test_what?"]      = "transformer"  
    
    # "deeppurpose"   
    arg["test_path"]       = "data/dataset_test.csv"                                             
    
    # "transformer"
    arg["test_path_t"]     = "./experiment/albumin/albumin_ligand/"      
    
    # "random"
    arg["test_path_r"]     = "./data/chembl_28_all.txt"                                           
    arg['test_r_output']   = "./result/translate_on_experimentset/random_on_all_chembl28/"
    
    main(arg)
