import numpy as np
from interface.BeamSearch import *
import tensorflow as tf


class NewsPredictor(Predictor):
    def __init__(self,estimator,topk = None):
        self.topk = topk
        self.estimator = estimator


    def predict(self,source,context,con_len,topk = None):
        def input_fn():
            return {'source':tf.constant(source),'context':tf.constant(context),'con_len':tf.constant(con_len)},tf.constant(0)
            # return tf.data.Dataset.from_generator(lambda :{'source':tf.constant(source),'context':tf.constant(context),'con_len':tf.constant(con_len)},tf.constant(0))

        pred = self.estimator.predict(input_fn,'target',yield_single_examples=False)
        res_v = []
        res = next(pred)
        for i,v in enumerate(res['target']):
            vmap = v[con_len]
            sort_res = np.argsort(vmap)[-topk:]
            map_res = {}
            for k in sort_res:
                map_res[k] = vmap[k]
            res_v.append(map_res)
        return res_v
        # count = len(res)


        # print(count)




class NewsBeamsearcher(Beamsearcher):




    def do_search_mt(self,max_step,estimator):
        # buffer_lock = threading.RLock()
        # # result_lock = threading.RLock()
        # buffer_lock = threading.Condition(1)
        # result_lock = threading.BoundedSemaphore(self.topk)

        def fill_data(searcher):
            while len(searcher.gen_result) < searcher.max_count:

                # result_lock.acquire()
                if len(searcher.buffer) == 0 or searcher.gen_len == max_step:

                    source = next(searcher.dataset)
                    searcher.source_input = source[:]
                    if searcher.gen_len == max_step:
                        searcher.gen_result.append(searcher.buffer)
                    searcher.buffer = []
                    for i in range(searcher.topk):
                        searcher.buffer.append(([],0))
                    searcher.gen_len = 0
                    searcher.next_topk = queue.Queue(1)
                    searcher.context = queue.Queue(1)
                else:
                    tmp_buffer = []
                    buffer = searcher.buffer
                    next_topk = searcher.next_topk.get()
                    for i, val in enumerate(buffer):
                        candidate,score = val
                        for n in next_topk[i]:
                            ts = next_topk[i][n]
                            tc = candidate[:]
                            tc.append(n)
                            tmp_buffer.append((tc,score+ts))

                    sorted(tmp_buffer,key=lambda x:x[1])
                    tmp_buffer = tmp_buffer[:searcher.topk]
                    searcher.buffer = tmp_buffer
                    searcher.gen_len += 1

                context = []
                print('第{0}步生成的内容：'.format(searcher.gen_len))
                for seq in self.buffer:
                    # con_len.append(len(seq[0]))
                    context.append(self.tokenizer.padding(seq[0][:],max_step))
                    print(searcher.tokenizer.get_sentence(seq[0][:]))

                searcher.context.put(context)


        def data_generator():
            searcher = self
            while len(searcher.gen_result) < searcher.max_count:
                context =  searcher.context.get()
                # buffer_lock.wait(2)
                for c in context:
                    yield {'source':tf.constant(searcher.source_input),'context':tf.constant(c)}, tf.constant(0)




        def input_fn():
            return tf.data.Dataset.from_generator(generator=data_generator,output_types=({'source':tf.int64,'context':tf.int64},tf.int64),output_shapes=({'source':[1000],'context':[100]},[])).batch(self.topk)

        pred = estimator.predict(input_fn,'target',yield_single_examples=False)

        def get_next(searcher):

            while len(searcher.gen_result) < searcher.max_count:
                # result_lock.acquire()
                res_v = []
                res = next(pred)
                # print('生成下一步')
                for i,v in enumerate(res['target']):
                    vmap = v[searcher.gen_len-1]
                    sort_res = np.argsort(vmap)[-searcher.topk:]
                    map_res = {}
                    for k in sort_res:
                        map_res[k] = vmap[k]
                    res_v.append(map_res)

                searcher.next_topk.put(res_v)
                # result_lock.notify()
                # buffer_lock.release()

        producer = threading.Thread(target=fill_data,args=(self,))
        consumer = threading.Thread(target=get_next,args=(self,))
        producer.start()
        consumer.start()

        producer.join()
        consumer.join()