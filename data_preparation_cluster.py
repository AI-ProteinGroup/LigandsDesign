import os
from typing import Iterator
import numpy as np
import pandas as pd
import subprocess
import random

def process_data(flag):

    # 从swissprot数据库中挑出bindingDB中满足条件的序列
    if flag["filter_swissprot"]:
        # 取bindingDB得到的数据集的所有蛋白的id
        train_set     = pd.read_csv("data/proteins_and_ligands_train.csv")
        test_set      = pd.read_csv("data/proteins_and_ligands_test.csv")
        valid_set     = pd.read_csv("data/proteins_and_ligands_valid.csv")
        dataset       = pd.concat(objs = [train_set, valid_set, test_set], axis = 0)
        my_protein    = set(dataset['UniProt (SwissProt) Primary ID of Target Chain'])
        
        # 去除swissprot中不是bindingDB中的序列
        uniprot_file  = open("data/uniprot_swissprot.fasta","r")
        filtered_file = open("data/uniprot_my.fasta","a")
        write_flag = False

        # 遍历每行，当一行以字符串 '>sp|'开头时，表示一条数据的第一行。 
        # 取其蛋白id，如果符合条件，则write_flag置True,整条数据都会被写入。
        for line in uniprot_file:
            if line.startswith('>sp|') or line.startswith('>tr|'):
                if line.startswith('>sp|'):
                    name = line.replace('>sp|', '').split(' ')[0].split('|')[0]
                else :
                    name = line.replace('>tr|', '').split(' ')[0].split('|')[0]
                
                if name in my_protein:
                    write_flag = True
                else:
                    write_flag = False
            
            if write_flag:
                filtered_file.write(line)
            
        uniprot_file.close()
        filtered_file.close()

    # mmseqs对bindingDB中的序列分族
    if flag["cluster_sequence"]:
        process = subprocess.Popen("mmseqs easy-cluster data/uniprot_my.fasta data/clusterRes data/tmp", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        process.wait()
        print(process.stdout.read())

    # 汇总分族完的蛋白
    if flag["process_clusters"]:
        cluster_files = pd.read_csv('data/clusterRes_cluster.tsv', sep='\t',header=None, names=['res_id','seq_id'])
        seq_is_in = dict()       # seq_is_in  key: seq_id    val:  res_id
        
        # tsv数据汇总到字典中
        for (res_id, seq_id) in zip(cluster_files['res_id'], cluster_files['seq_id']):
            seq_is_in[seq_id] = res_id
        
        # 读取csv文件，整合到dataset中
        train_set     = pd.read_csv("data/proteins_and_ligands_train.csv")
        test_set      = pd.read_csv("data/proteins_and_ligands_test.csv")
        valid_set     = pd.read_csv("data/proteins_and_ligands_valid.csv")
        dataset       = pd.concat(objs = [train_set, valid_set, test_set], axis = 0)

        # 发现bindingDB中部分数据pair的smiles包含多条，这里全部去掉
        def multiple_smiles(smi):
            if len(smi.strip().split(' ')) > 1:
                return True
            
            return False
        
        dataset["multiple_smi"] = dataset['Ligand SMILES'].apply(multiple_smiles)
        dataset = dataset.loc[dataset["multiple_smi"] == False, :]
        pair_num      = len(dataset['Ligand SMILES'])
        train_size    = int(0.8 * pair_num)
        valid_size    = int(0.1 * pair_num)

        # 清空三个表
        train_set     = train_set.drop(index=train_set.index)
        valid_set     = valid_set.drop(index=valid_set.index)
        test_set      = test_set.drop(index=test_set.index)

        # clusters    key: res_id    val: list of pairs
        clusters = dict()       
        for index, row in dataset.iterrows(): 
            seq_id = row['UniProt (SwissProt) Primary ID of Target Chain']

            if seq_id not in seq_is_in:
                continue

            res_id = seq_is_in[seq_id]
            
            if res_id in clusters.keys():
                pass
            else:
                clusters[res_id] = []

            clusters[res_id].append({
                'UniProt (SwissProt) Primary ID of Target Chain' : row['UniProt (SwissProt) Primary ID of Target Chain'],
                'BindingDB Target Chain  Sequence'               : row['BindingDB Target Chain  Sequence'],
                'ChEMBL ID of Ligand'                            : row['ChEMBL ID of Ligand'],
                'Ligand SMILES'                                  : row['Ligand SMILES'],
            })
        
        # 放入df中
        num = 0
        for cluster in clusters.values():
            if num < train_size:
                train_set = train_set.append(cluster, ignore_index=True)
            elif num < train_size + valid_size:
                valid_set = valid_set.append(cluster, ignore_index=True)
            else:
                test_set  = test_set.append(cluster, ignore_index=True)

            num += len(cluster)

        train_set.to_csv("data/dataset_train.csv", columns= ['UniProt (SwissProt) Primary ID of Target Chain','BindingDB Target Chain  Sequence','ChEMBL ID of Ligand','Ligand SMILES'])
        valid_set.to_csv("data/dataset_valid.csv", columns= ['UniProt (SwissProt) Primary ID of Target Chain','BindingDB Target Chain  Sequence','ChEMBL ID of Ligand','Ligand SMILES'])
        test_set.to_csv("data/dataset_test.csv",   columns= ['UniProt (SwissProt) Primary ID of Target Chain','BindingDB Target Chain  Sequence','ChEMBL ID of Ligand','Ligand SMILES'])
                

if __name__ == "__main__":

    flag = dict()
    flag["filter_swissprot"] = False   # uniprot_swissprot.fasta中挑出bindingDB中的蛋白序列，保存到uniprot_my.fasta中
    flag["cluster_sequence"] = False   # 分族交给软件mmseqs去做
    flag["process_clusters"] = True    # 分族后的蛋白id存在 clusterRes_cluster.tsv中。

    process_data(flag)