import random
from sys import maxsize
import torch  
import copy

# 用于seq2seq训练transformer的的数据集
class PairDataset(torch.utils.data.Dataset):
    """Some Information about PairDataset"""
    def __init__(self, data):
        super(PairDataset, self).__init__()

        # 两个长度相同的list
        self.drugs = data['drugs']           
        self.proteins = data['proteins']
        assert len(self.drugs) == len(self.proteins)

        # 求最长的smiles的长度
        maxlen = 0
        for d in self.drugs:
            maxlen = max(maxlen, len(d))
        self.max_len_drugs = maxlen 

        # 求最长的氨基酸序列的长度
        maxlen = 0
        for d in self.proteins:
            maxlen = max(maxlen, len(d))
        self.max_len_proteins = maxlen

    def __getitem__(self, index):
        return { 'protein' : self.proteins[index], 'drug' : self.drugs[index]}

    def __len__(self):
        return len(self.drugs)

    def get_max_len(self):
        return {'drug':self.max_len_drugs, 'protein':self.max_len_proteins}

    staticmethod
    def collate_fn(batch):
        drugs = [torch.LongTensor(p['drug']) for p in batch]
        proteins = [torch.LongTensor(p['protein']) for p in batch]

        # padding对齐
        drugs = torch.LongTensor(torch.nn.utils.rnn.pad_sequence(drugs, batch_first=True, padding_value=0))
        proteins = torch.LongTensor(torch.nn.utils.rnn.pad_sequence(proteins, batch_first=True, padding_value=0))
        return {
            'drugs'    : drugs,
            'proteins' : proteins
        }

# 用于bert预训练的数据集
class MonoDataset(torch.utils.data.Dataset):
    def __init__(self, seqs, vocab) -> None:
        super().__init__()
        self.seqs = seqs
        self.vocab = vocab

        maxlen = 0
        for s in seqs:
            maxlen = max(maxlen, len(s))
        self.maxlen = maxlen

    def get_max_len(self):
        return self.maxlen

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        seq = copy.deepcopy(self.seqs[index]) 
        label = []

        # 参考BERT的mask language model方法
        # 这里我并没有用到Next sentence prediction方法。
        # 但是protbert他们好像用到了，所以论文里我还是介绍了。
        for i, token in enumerate(seq):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                if prob < 0.8:
                    seq[i] = 3 # 替换为mask的index
                
                elif prob < 0.9:
                    seq[i] = random.randrange(6, self.vocab.get_size())   # 随机替换
                
                else:
                    pass       # 不变
                
                label.append(token)
            else:
                label.append(0)  # 计算loss时，0会被忽略
        
        return {               # 生成的label，用于计算bert loss
            'seq': seq,
            'label' : label
        }

    staticmethod
    def collate_fn(batch):
        seqs = [torch.LongTensor(p['seq']) for p in batch]
        labels = [torch.LongTensor(p['label']) for p in batch]
        seqs = torch.LongTensor(torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0))
        labels = torch.LongTensor(torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0))
        return {
            'seqs': seqs,
            'labels' : labels
        }

# 用于seq2seq训练protbert的数据集
class ProtransDataset(torch.utils.data.Dataset):
    """Some Information about PairDataset"""
    def __init__(self, data, protein_vocab, device):
        super(ProtransDataset, self).__init__()
        self.drugs = data['drugs']
        self.proteins = data['proteins']
        assert len(self.drugs) == len(self.proteins)

        maxlen = 0
        for d in self.drugs:
            maxlen = max(maxlen, len(d))
        self.max_len_drugs = maxlen

        maxlen = 0
        for d in self.proteins:
            maxlen = max(maxlen, len(d))
        self.max_len_proteins = maxlen

        proteins = []
        for protein in self.proteins:
            p = protein[1:-1]
            p = ' '.join([ protein_vocab.to_word(index) for index in p])
            proteins.append(p)

        self.proteins = proteins


    def __getitem__(self, index):
        return { 'protein' : self.proteins[index], 'drug' : self.drugs[index]}

    def __len__(self):
        return len(self.drugs)

    def get_max_len(self):
        return {'drug':self.max_len_drugs, 'protein':self.max_len_proteins}

    staticmethod
    def collate_fn(batch):
        drugs = [torch.LongTensor(p['drug']) for p in batch]
        drugs = torch.LongTensor(torch.nn.utils.rnn.pad_sequence(drugs, batch_first=True, padding_value=0))

        proteins = [item['protein'] for item in batch]

        return {
            'drugs'    : drugs,
            'proteins' : proteins
        }