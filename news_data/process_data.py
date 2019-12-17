import sys
sys.path.append("/home/user/zsm/Summarization/")
from data_util.data_source import NewsDatasetBuilder




if __name__ == '__main__':
    # l = NewsDatasetBuilder(open("/home/data/news2016.json",'r',encoding='utf-8'), "NEWS")
    # l.build_dataset()
    dict_f = open('NEWS_DICT.txt','r',encoding='utf-8')
    single_dict_f = open('NEWS_DICT_SINGLE.txt','w',encoding='utf-8')

    re_dict_f = open('NEWS_DICT_R.txt','w',encoding='utf-8')

    single_d = []
    re_d = []
    for line in dict_f:
        res = line.strip().split(' ')
        try:
            if len(res[1]) == 1:
                single_d.append(res[1:])
            else:
                re_d.append(res[1:])
        except Exception:
            continue
    print('读取完毕 单字{}个，词汇表{}个'.format(len(single_d),len(re_d)))
    c = 2
    single_dict_f.write('0 <PAD> x 0\n1 <EOS> x 0\n')
    re_dict_f.write('0 <PAD> x 0\n1 <EOS> x 0\n')
    for w in single_d:
        single_dict_f.write("{} {} {} {}\n".format(c,w[0],w[1],w[2]))
        re_dict_f.write("{} {} {} {}\n".format(c,w[0],w[1],w[2]))
        c += 1
    for w in re_d:
        re_dict_f.write("{} {} {} {}\n".format(c,w[0],w[1],w[2]))

        c += 1
        if c>=100000:
            break

