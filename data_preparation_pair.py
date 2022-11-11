import pandas as pd
import numpy as np 
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
import re
import random 
import os

# 读取文件
# 由于读取的是TSV文件，分隔符用的是'\t'
print('\n读取文件......')
data = pd.read_csv('data/BindingDB_All.tsv', sep = '\t', error_bad_lines=False)
print("file shape:")
print(data.shape)

# 提取规则1：人类靶蛋白
# 通过逻辑运算来提取数据子集
print('\n按照规则1：蛋白为人类靶蛋白 进行筛选......')
data = data.loc[data['Target Source Organism According to Curator or DataSource'] == 'Homo sapiens',:]
print("data shape:")
print(data.shape)

# 提取规则2：IC50 EC50 KD KI < 10000nm (pxc > 5)     
# 这几列都是字符串，  可能的数据是  数值，或者>数值,<数值。 
# 因此需要字符串进行分类处理
print('\n按照规则2：IC50 EC50 KD <10000nm (pxc > 5) 进行筛选......')
bad_val = []
def convert_to_float(val, allowed_prefixes=('>', '<')):
    if isinstance(val, (int, float)):
        return float(val)
    
    val = str(val)
    if val[0] not in allowed_prefixes:
        #raise Exception(f'Conversion failed! Got "{val}"')
        bad_val.append(val)
        return 50000
        
    return float(val[1:])

def process_signs(val):
    val = str(val)
    if val[0]=='>':
        return 'greater'
    else:
        return 'else'


allowed_prefixes=('>', '<', ' ') # allowed_prefixes=('>', '<') # will fail, since some items start with space

tempdata = data

tempdata['IC50_signs'] = data['IC50 (nM)'].apply(process_signs)
tempdata['IC50_floats_data'] = data['IC50 (nM)'].apply(convert_to_float, allowed_prefixes=allowed_prefixes)
tempdata['IC50_nan'] = pd.isnull(tempdata['IC50_floats_data'])

tempdata['Kd_signs'] = data['Kd (nM)'].apply(process_signs)
tempdata['Kd_floats_data'] = data['Kd (nM)'].apply(convert_to_float, allowed_prefixes=allowed_prefixes)
tempdata['Kd_nan'] = pd.isnull(tempdata['Kd_floats_data'])

tempdata['EC50_signs'] = data['EC50 (nM)'].apply(process_signs)
tempdata['EC50_floats_data'] = data['EC50 (nM)'].apply(convert_to_float, allowed_prefixes=allowed_prefixes)
tempdata['EC50_nan'] = pd.isnull(tempdata['EC50_floats_data'])

tempdata['Ki_signs'] = data['Ki (nM)'].apply(process_signs)
tempdata['Ki_floats_data'] = data['Ki (nM)'].apply(convert_to_float, allowed_prefixes=allowed_prefixes)
tempdata['Ki_nan'] = pd.isnull(tempdata['Ki_floats_data'])

tempdata.loc[(tempdata['IC50_nan'] == False) & (tempdata['IC50_floats_data'] < 10000) & (tempdata['IC50_signs'] != 'greater'), 'IC50_result']=True
tempdata.loc[(tempdata['Kd_nan'] == False) & (tempdata['Kd_floats_data'] < 10000) & (tempdata['Kd_signs'] != 'greater'), 'Kd_result']=True
tempdata.loc[(tempdata['EC50_nan'] == False) & (tempdata['EC50_floats_data'] < 10000) & (tempdata['EC50_signs'] != 'greater'), 'EC50_result']=True
tempdata.loc[(tempdata['Ki_nan'] == False) & (tempdata['Ki_floats_data'] < 10000) & (tempdata['Ki_signs'] != 'greater'), 'Ki_result']=True


result = tempdata.loc[(tempdata['Ki_result'] == True) | ((tempdata['IC50_result'] == True) & (tempdata['Ki_nan']==True)) | ((tempdata['Ki_nan'] == True) & (tempdata['IC50_nan']==True) & (tempdata['Kd_result'] == True)) | ((tempdata['Ki_nan'] == True) & (tempdata['IC50_nan']==True) & (tempdata['Kd_nan'] == True) & (tempdata['EC50_result'] == True) )]

print("data shape:")
print(result.shape)
print("\nbad value num:")
print(len(bad_val))

# 提取规则4:必须有配体的smiles
print('\n按照规则4:必须有配体的smiles  进行筛选......')
result2 = result.loc[pd.isnull(result['Ligand SMILES']) == False]
print("data shape:")
print(result2.shape)


# 提取规则3:smiles必须有PubChem CID
print('\n按照规则:smiles必须有PubChem CID  进行筛选......')
with_pubchem_cid = result2.loc[pd.isnull(result2['PubChem CID']) == False]
print("data shape:")
print(with_pubchem_cid.shape)

# 提取规则5:分子重量小于500
print('\n按照规则:分子重量小于500  进行筛选......')

def mol_weight(smile):
  try:
    return Descriptors.MolWt(Chem.MolFromSmiles(str(smile)))
  except Exception:
    return np.nan

with_pubchem_cid['mol_weight'] = with_pubchem_cid['Ligand SMILES'].apply(mol_weight)
with_mol_weight_smile = with_pubchem_cid.loc[(pd.isnull(with_pubchem_cid['mol_weight'])==False) & (with_pubchem_cid['mol_weight'] < 501)]
print("data shape:")
print(with_mol_weight_smile.shape)

# 提取规则6、7      蛋白质序列有UniprotID    并且长度位于80-2050
print('\n按照规则6、7:蛋白质序列有UniprotID    并且长度位于80-2050  进行筛选......')

def process_protein(ds_to_start):
    var_name1 = 'BindingDB Target Chain  Sequence'
    var_name2 = 'UniProt (SwissProt) Primary ID of Target Chain'
    prot_cond = ds_to_start.loc[pd.isnull(ds_to_start[var_name1]) == False ]
    prot_cond = prot_cond.loc[pd.isnull(prot_cond[var_name2]) == False]
    print('shape_before_len ', prot_cond.shape)
    if prot_cond.empty == False:
        prot_cond2 = prot_cond.loc[((prot_cond[var_name1].str.len() > 79) & (prot_cond[var_name1].str.len() < 2049))]
        print('shape_after_len ',prot_cond2.shape)
        return prot_cond2
    else:
        return prot_cond

# 提取规则8 单链
prot_cond0 = with_mol_weight_smile.loc[with_mol_weight_smile['Number of Protein Chains in Target (>1 implies a multichain complex)'] == 1]
print('\n处理目标链等于1的情况')
prot_1_chain = process_protein(ds_to_start = prot_cond0)
print('prot_1_chain:', prot_1_chain.shape)

# 保存所有单链pair数据
ids=[]
with open('data/proteins_and_ligands_all.csv', 'w+') as prot_file:
  prot_1_chain[['UniProt (SwissProt) Primary ID of Target Chain','BindingDB Target Chain  Sequence','ChEMBL ID of Ligand', 'Ligand SMILES']].to_csv(prot_file)  
  ids = list(set(list(prot_1_chain['BindingDB Target Chain  Sequence'])))
  smis = list(set(list(prot_1_chain['Ligand SMILES'])))

# 保存所有单链pair数据和其他信息
with open('data/proteins_and_ligands_full.csv', "w+") as t:
    prot_1_chain.to_csv(t)

print('\n最终数据:')
print("pair num:")
print(prot_1_chain.shape)           # prot_1_chain 是所有 protein-smiles pairs 的pandas。 pair的序列数据存在proteins_and_ligands.csv , 其他详细信息存在dataset_full.csv
print("protein num:")
print(len(set(ids)))                # ids 是所有蛋白链的unique set
print("mol num:")
print(len(set(smis)))                # ids 是所有蛋白链的unique set

##################################################  8:1:1 划分数据集  ##################################################

# 排除 drd2 5ht2a jak2蛋白作为最后实验验证
experiment_list  = ["A0A024R3C5","P13953","P14416","P20288","P52702","P61168","P61169","Q0VGH9","Q9GJU1","Q9NZR3","Q9UPA9", "A0A4X1T900","F1SM69","G1T1Q0","Q589Y6"] # DRD2蛋白的id  P14416
experiment_list += ["A0A024RD15","A6ZJ92","A8K6P4","B0LPH0","B2RAC5","B4DZ79","B5TY32","F5GWE8","O46635","O60776","P14842","P28223","P35363","Q13083","Q14084","Q16539","Q50DZ9","Q543D4","Q5T8C0","Q60F96","Q8TDX0"] # 5HT2A蛋白的id Q16539
experiment_list += ["O14636","O60674","O75297","Q62120","Q62124","Q7TQD0","Q506Q0"] # jak2蛋白的id O60674
experiment_list = set(experiment_list)

def is_experiment_list(protein_id):
    if protein_id in experiment_list:
        return "experimentset"
    else:
        return ""

prot_1_chain["temp1_data_type"] =  prot_1_chain["UniProt (SwissProt) Primary ID of Target Chain"].apply(is_experiment_list)
prot_1_chain["temp2_data_type"] =  prot_1_chain["UniProt (TrEMBL) Primary ID of Target Chain"].apply(is_experiment_list)
prot_1_chain.loc[prot_1_chain["temp1_data_type"]=="experimentset", "data_type"] = "experimentset"
prot_1_chain.loc[prot_1_chain["temp2_data_type"]=="experimentset", "data_type"] = "experimentset"

print("\n DRD2, 5HT2A, JAK2的数据对：")
print(prot_1_chain.loc[prot_1_chain["data_type"] == "experimentset"])

# 其他的蛋白链
ids = list(set(prot_1_chain.loc[prot_1_chain["data_type"] != "experimentset"]['BindingDB Target Chain  Sequence']))

random.shuffle(ids)
train_len = int(0.8 * len(ids))
valid_len = int(0.9 * len(ids))

train_proteins = set(ids[:train_len])
valid_proteins = set(ids[train_len:valid_len])
test_proteins  = set(ids[valid_len:])

def dataset_type(protein_string):
    if protein_string in train_proteins:
        return "trainset"
    elif protein_string in valid_proteins:
        return "validset"
    elif protein_string in test_proteins:
        return "testset"
    else:
        return "experimentset"

prot_1_chain["data_type"] = prot_1_chain['BindingDB Target Chain  Sequence'].apply(dataset_type)

with open("data/proteins_and_ligands_test.csv","w+") as test_file, open("data/proteins_and_ligands_valid.csv","w+") as valid_file, open("data/proteins_and_ligands_train.csv","w+") as train_file, open("data/proteins_and_ligands_experiment.csv","w+") as experiment_file:
    prot_1_chain.loc[prot_1_chain["data_type"] == "trainset"][['UniProt (SwissProt) Primary ID of Target Chain', 'BindingDB Target Chain  Sequence', 'ChEMBL ID of Ligand','Ligand SMILES']].to_csv(train_file)
    prot_1_chain.loc[prot_1_chain["data_type"] == "validset"][['UniProt (SwissProt) Primary ID of Target Chain', 'BindingDB Target Chain  Sequence', 'ChEMBL ID of Ligand','Ligand SMILES']].to_csv(valid_file)
    prot_1_chain.loc[prot_1_chain["data_type"] == "testset"][['UniProt (SwissProt) Primary ID of Target Chain', 'BindingDB Target Chain  Sequence', 'ChEMBL ID of Ligand','Ligand SMILES']].to_csv(test_file)
    prot_1_chain.loc[prot_1_chain["data_type"] == "experimentset"][['UniProt (SwissProt) Primary ID of Target Chain', 'BindingDB Target Chain  Sequence', 'ChEMBL ID of Ligand','Ligand SMILES']].to_csv(experiment_file)