import os
import pickle
import string
from tqdm import tqdm

padToken, goToken, eosToken, unknownToken = 0, 1, 2, 3
dataset_filename = 'data/dataset-70425-vocabSize33050.pkl'


def loadDataset(filename):
    '''
    读取样本数据
    :param filename: 文件路径，是一个字典，包含word2id、id2word分别是单词与索引对应的字典和反序字典，
                    trainingSamples样本数据，每一条都是QA对
    :return: word2id, id2word, trainingSamples
    '''
    dataset_path = os.path.join(filename)
    print('Loading dataset from {}'.format(dataset_path))
    with open(dataset_path, 'rb') as handle:
        data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
        word2id = data['word2id']
        id2word = data['id2word']
        trainingSamples = data['trainingSamples']
    return word2id, id2word, trainingSamples


def sequence2Str(sequence, id2word, clean=False, reverse=False):
    if not sequence:
        return ''
    if not clean:
        return ''.join([id2word[idx] for idx in sequence])
    sentence = []
    for idx in sequence:
        if idx == eosToken:
            break
        elif idx != goToken and idx != padToken:
            if idx in id2word:
                sentence.append(id2word[idx])
            else:
                sentence.append('<unk>')
    if reverse:
        sentence.reverse()

    return detokenize(sentence)


def detokenize(tokens):
    """Slightly cleaner version of joining with spaces.
    Args:
        tokens (list<string>): the sentence to print
    Return:
        str: the sentence
    """
    return ''.join([
        ' ' + t if not t.startswith('\'') and
                   t not in string.punctuation
        else t for t in tokens]).strip().capitalize()


def build_dataset():
    word2id, id2word, trainingSamples = loadDataset(dataset_filename)
    file_post = open('data/train_Q.post', 'w', encoding='utf-8')
    file_resp = open('data/train_Q.response', 'w', encoding='utf-8')

    for sample in tqdm(trainingSamples,desc='Building Dataset'):
        post = sequence2Str(sample[0], id2word, clean=True) + '\n'
        response = sequence2Str(sample[1], id2word, clean=True) + '\n'
        file_post.write(post)
        file_resp.write(response)

    file_post.close()
    file_resp.close()


if __name__ == '__main__':
    build_dataset()
