import pandas as pd

def get_init_dict() -> dict:
    ret = {}
    ret['pad'] = 0
    ret['bos'] = 1
    ret['eos'] = 2
    ret['mask'] = 3
    ret['remain_1'] = 4   # 预留出来的位置，后续可能有任务可以用到。
    ret['remain_2'] = 5   # 预留出来的位置，后续可能有任务可以用到。
    return ret

class Vocabulary(object):
    def __init__(
                self,
                vocab_path,
                ) -> None:

        self.word_to_index = get_init_dict()
        next_index = len(self.word_to_index)
        
        with open(vocab_path) as file:
            lines = file.readlines()
            for line in lines:
                self.word_to_index[line.strip()] = next_index
                next_index += 1

        self.index_to_word = {}
        for word in self.word_to_index:
            index = self.word_to_index[word]
            self.index_to_word[index] = word
    
    def to_word(self, index:int) -> str:
        return self.index_to_word[index]

    def to_index(self, word:str) -> int:
        return self.word_to_index[word]

    def get_size(self) -> int:
        return len(self.word_to_index)

if __name__ == '__main__':
    pass