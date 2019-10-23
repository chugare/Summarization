import redis
import pickle
import time
import random
r = redis.Redis(host='localhost', port=6379, db=0)

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
    result = {}
    for k in keys:
        result[k] = None
    keys = list(result.keys())

    st = time.time()
    for i in keys:
        pipe.get(str(i))
    res = pipe.execute()
    for i, rr in enumerate(res):
        if not rr:
            print('[ERROR] Key %s get fail' % keys[i])
        else:
            result[keys[i]] = pickle.loads(rr)
    dt = time.time()-st
    print(dt)
    return result
