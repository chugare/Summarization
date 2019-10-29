import redis
import pickle
import time
import random
r = redis.Redis(host='localhost', port=6379, db=0)
r_miss = redis.Redis(host='localhost', port=6379, db=1)
miss_key = {}

def init():
    r.flushdb()

def save_value(kv):
    pipe = r.pipeline(transaction=False)
    st = time.time()
    keys = list(kv.keys())
    for k in kv:
        out_s = pickle.dumps(kv[k])
        pipe.set(k,out_s)
    res = pipe.execute()
    for i,rr in enumerate(res):
        if not rr:
            print('[ERROR] Key %s inserted fail' % keys[i])
    et = time.time()
    dt = et-st
def get_value(keys):
    pipe = r.pipeline()
    # miss_pipe = r_miss.pipeline()
    result = {}
    st = time.time()
    for i in keys:
        pipe.get(str(i))
    res = pipe.execute()
    for i, rr in enumerate(res):
        if not rr:
            miss_key[keys[i]] = miss_key.get(keys[i],0)+1
        else:
            result[keys[i]] = pickle.loads(rr)
    # dt = time.time()-st
    return result

