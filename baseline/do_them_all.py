from baseline.Lead_n import covage,lead
from baseline.TextRank import textRank
from baseline.Tf_idf import do_tfidf
from baseline.util import mkreport

DICT_PATH = '/root/zsm/Summarization/news_data_r/NEWS_DICT_R.txt'
DATA_PATH = '/home/user/zsm/Summarization/news_data/NEWS.txt'
E_DATA_PATH = '/home/user/zsm/Summarization/news_data/E_NEWS.txt'



fun_map = {
    'lead':lead,
    'covage':covage,
    'tfidf':do_tfidf,
    'textrank':textRank,
}
for name in fun_map:
    mkreport(fun_map[name],name,DATA_PATH,400)
# mkreport(covage,'covage',DATA_PATH,400)



