import os
import re
file = open('DP.txt','r',encoding='utf-8')
ofile = open('DP50.txt','w',encoding='utf-8')
c = 0

for line in file:
    patterns = [
        r"[（\(]+[一二三四五六七八九十\d]+[\)）]+[，、。．,\s]*",
        r"[⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽⑾⑿⒀⒁⒂⒃⒄⒅⒆⒇]",
        # r"[\s]",
        r"[a-zA-Z《》【】（）!@#$%^&*\s]+",
    ]
    for p in patterns:
        line = re.sub(p, '', line)
    words = line.split(' ')
    if len(words)>50 and len(words)<100:


        c+= 1
        ofile.write(line.strip()+'\n')
