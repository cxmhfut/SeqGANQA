import re

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3


class Reader:
    def __init__(self, file_name_post, file_name_resp, file_name_word):
        with open(file_name_word, 'r', encoding='utf-8') as file_word:
            self.d = {}
            self.symbol = []
            num = 0
            for line in file_word.readlines():
                line = line[:-1]
                self.symbol.append(line)
                self.d[line] = num
                num += 1

        self.file_name_post = file_name_post
        self.file_name_resp = file_name_resp
        self.post = open(self.file_name_post, 'r', encoding='utf-8')
        self.resp = open(self.file_name_resp, 'r', encoding='utf-8')
        self.epoch = 0
        self.k = 0

    def get_batch(self, batch_size):
        result = []
        self.k += batch_size
        for _ in range(batch_size):
            post = self.post.readline()
            resp = self.resp.readline()
            if not post:
                self.restore()
                self.epoch += 1
                self.k = 0
            post = post[:-1]
            resp = resp[:-1]
            words_post = re.split(r' ', post)
            words_resp = re.split(r' ', resp)
            index_post = [self.d[word] if word in self.d else UNK_ID for word in words_post]
            index_resp = [self.d[word] if word in self.d else UNK_ID for word in words_resp]
            result.append((index_post, index_resp))
        return result

    def restore(self):
        self.post.close()
        self.resp.close()
        self.post = open(self.file_name_post, 'r', encoding='utf-8')
        self.resp = open(self.file_name_resp, 'r', encoding='utf-8')

if __name__ == '__main__':
    reader = Reader('data/train_Q.post',
                    'data/train_Q.response',
                    'data/result.txt')

    result = reader.get_batch(10)
    for item in result:
        print(item)