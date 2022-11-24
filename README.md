# De Novo Design of Target-Specific Ligands Using BERT-Pretrained Transformer
<center>
<img src="./png/Figure 1.png"/>

<img src="./png/Figure 2.png"/>
</center>

## Requirements
We think the requirements for the environment are so loose that any version can be used

python=3.8.12
pytorch=1.9.1+cu111
transformers=4.12.5
numpy=1.21.2
pandas=1.3.3
dill=0.3.4
tqdm=4.62.3
rdkit=2021.09.2

## Train Transformer-ED
First train transformer's decoder:
```python
python protrans_pretrain.py
```
then fineturn the transformer:
```python
python protrans_finetune.py
```
The model is saved in "save_folder/protrans_finetune/protrans_e_bert_d_transformer_token_token_l0.00001_e30/"

You can modify "$arg['output\_dir']$" to change the location of the saved weight file

## generate molecules

Use the trained Transformer-ED to generate small molecules, you can use our [weight_file](https://drive.google.com/drive/folders/1Fy5ye0ndolcWubEDJqx5QVVGbamPd-3H?usp=share_link) directly

```python
python protrans_mose.py
```
the output in "result/translate_on_experimentset/protrans_e_bert_d_transformer_token_token_l0.00001_e30_random100/generated.csv"

## Data
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow" colspan="1">data_name</th>
    <th class="tg-c3ow" colspan="1">path</th>
	<th class="tg-c3ow" colspan="1">description</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow">Binding DB</td>
    <td class="tg-c3ow">data/dataset_train.csv<br>data/dataset_valid.csv<br>data/dataset_test.csv</td>
	<td class="tg-c3ow">Pairwise data of protein sequence and molecular sequence</td>
  </tr>
  
  <tr>
    <td class="tg-c3ow">ChEMBL</td>
    <td class="tg-c3ow">data/chembl_28_pxc_5.txt </td>
    <td class="tg-c3ow">Active molecules were selected for PXC > 5 molecules</td>
  </tr>
    <tr>
    <td class="tg-c3ow">Uniport/Swissport</td>
	<td class="tg-c3ow">data/uniprot_human_all.xlsx</td>
    <td class="tg-c3ow">Human protein Data</td>
  </tr>
</tbody>
</table>

All the above data can be found [here](https://drive.google.com/drive/folders/1owQfTuer67qlN3OG0xtFSJzy_ebOIuTu?usp=share_link)

## Docking Results 
<center>
<img src="./png/Figure 4.png"/>

Fig. 4 Binding conformation of Risperidone and Rank 4 molecule docking with DRD2.

<img src="./png/Figure 5.png"/>

Fig. 5 Binding conformation of Lorpiprazole and Rank 3 molecule docking with 5HT2A.

<img src="./png/Figure 6.png"/>

Fig. 6 Binding conformation of Ruxolitinib and Rank 2 molecule docking with JAK2.
</center>

## model performance evaluation

<center>
<img src="./png/Figure 3.png"/>
</center>

## tode
Using DeepPurpose to predict the active part of the molecule will be given in the future.