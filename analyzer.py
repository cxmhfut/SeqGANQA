import re

file_post = open('data/train_Q.post', 'r')
file_resp = open('data/train_Q.response', 'r')

symbol = {}
for line in file_post.readlines():
    line = line[:-1]
    for word in re.split(r' ', line):
        if word in symbol:
            symbol[word] += 1
        else:
            symbol[word] = 1

for line in file_resp.readlines():
    line = line[:-1]
    for word in re.split(r' ', line):
        if word in symbol:
            symbol[word] += 1
        else:
            symbol[word] = 1

file_result = open('result.txt', 'w')

total = sum([x[1] for x in symbol.items()])
num = 0
word_num = 0

file_result.write('<go>\n<eos>\n<unk>\n<pad>\n')

for item in sorted(symbol.items(), key=lambda x: x[1], reverse=True):
    print(str([item[0].encode('utf-8')]))
    file_result.write(item[0] + '\n')
    num += item[1]
    word_num += 1
    if num >= total * 0.99:
        break

print(word_num)
