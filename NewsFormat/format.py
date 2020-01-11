from util.mysql_utils import connect_db
from util.redis_util import get_value
import jieba
import time
def get_all(batch_size = 10000):
    con = connect_db()
    cur = con.cursor()

    # res = None
    try:
        finish = False
        for i in range(100):
            cur.execute("select * from news_obj limit %d,%d"%(i * batch_size,batch_size))
            res = cur.fetchall()
            yield res
            con.commit()
    except Exception as e:
        print("[W] some data insert failed, but process continue")
        raise e
    finally:
        cur.close()
        con.close()
    # return res


all_data = get_all()

st = time.time()
lt = st
count = 0
for ad in all_data:
    # print(ad)
    count += 10000
    nt = time.time()
    print('read %d lines, cost %.2f time ,this batch cost %.2f'%(count,nt-st,nt-lt))
    lt = nt


