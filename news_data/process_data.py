import sys
sys.path.append("/home/user/zsm/Summarization/")
# from data_util.news_2016.txt_builder import NewsDatasetBuilder
from data_util.news_2016.mysql_builder import NewsDatasetMysqlWriter
from data_util.news_2016.txt_builder import NewsDatasetFromMysql


def  mk_txt():
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





if __name__ == '__main__':
    # l = NewsDatasetBuilder(open("/home/data/news2016.json",'r',encoding='utf-8'), "NEWS")
    # l = NewsDatasetMysqlWriter(open("/home/data/news2016.json",'r',encoding='utf-8'), "NEWS")
    # l.write_mysql()
    # l.build_dataset()
    # source = ['中国新闻网','新华网','光明网','京华时报']
    source = ['主流媒体-媒体平台']
    nd = NewsDatasetFromMysql(source)
    nd.build()
    # #
    # import jieba
    # rf = open('NEWS_TRK.txt','r',encoding='utf-8')
    # ttf = open('NEWS_TRKT.txt','w',encoding='utf-8')
    #
    # for line in rf:
    #     # try:
    #         r = line.split('#')
    #         if len(r) !=3:
    #             continue
    #         tr,s,content = r
    #         tr = ' '.join(jieba.cut(str(tr).replace(' ','')))
    #         ttf.write("%s#%s#%s\n"%(tr,s,content))
