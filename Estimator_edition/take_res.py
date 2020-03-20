f = open('reslog.txt','r',encoding='utf-8')
for line in f:

    l = line.strip().split('\t')[1:]

    print('\t'.join(l))