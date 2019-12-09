
import sys
sys.path.append('/home/user/zsm/Summarization')
import tensorflow as tf
import numpy as np
from data_util.tokenization import tokenization
from util.file_utils import queue_reader
from model.transformer import Transformer,create_look_ahead_mask,create_padding_mask
import time
import threading

from interface.BeamSearch import Predictor



def create_masks(inp, tar):
    # 编码器填充遮挡
    enc_padding_mask = create_padding_mask(inp)

    # 在解码器的第二个注意力模块使用。
    # 该填充遮挡用于遮挡编码器的输出。
    dec_padding_mask = create_padding_mask(inp)

    # 在解码器的第一个注意力模块使用。
    # 用于填充（pad）和遮挡（mask）解码器获取到的输入的后续标记（future tokens）。
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

def build_input_fn(name,data_set,batch_size = 32):

    def generator():
        tokenizer = tokenization("/root/zsm/Summarization/news_data/NEWS_DICT.txt",DictSize=100000)
        source_file = queue_reader(name,data_set)
        for line in source_file:
            try:
                example = line.split('#')
                title = example[0]
                desc = example[1]
                content = example[2]
                title = ''.join(title)
                content = ''.join(content)
                source_sequence = tokenizer.tokenize(content)
                source_sequence = tokenizer.padding(source_sequence,1000)
                title_sequence = tokenizer.tokenize(title)
                title_sequence = tokenizer.padding(title_sequence,100)
            # for i,s in enumerate(title_sequence):
            #     label = s
            #     context = title_sequence[:i]
            # #
                feature = {
                    'source':source_sequence,
                    'context':title_sequence
                }
            #     yield feature,label
                yield feature,0
            except Exception:
                continue
    def input_fn():
        ds = tf.data.Dataset.from_generator(generator=generator,output_types=({'source':tf.int64,'context':tf.int64},tf.int64),output_shapes=({'source':[1000],'context':[100]},[]))
        ds = ds.shuffle(8192).batch(batch_size).cache().repeat()
        return ds
    return input_fn





def build_model_fn(lr = 0.01,num_layers=3,d_model=200,num_head=8,dff=512,input_vocab_size=100000,
                            target_vocab_size=100000,
                            pe_input=1000,pe_target=100):

    learning_rate = lr
    #
    # loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    #     from_logits=True, reduction='none')
    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        # loss_ = loss_object(real, pred)
        loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)
    def accuracy_function(real,pred):
        train_accuracy = tf.keras.metrics.sparse_categorical_accuracy(real, pred)
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        mask = tf.cast(mask, dtype=train_accuracy.dtype)

        count = tf.reduce_sum(mask)+1
        train_accuracy = tf.reduce_sum(train_accuracy * mask) / count
        return train_accuracy


    def model_fn(features,labels,mode,params=None):

        # source = features['source']
        # context = features['context']

        global_step = tf.compat.v1.train.get_or_create_global_step()
        source = features['source']
        context = features['context']
        tar_inp = context[:, :-1]
        tar_real = context[:, 1:]
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(source, tar_inp)

        class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
            def __init__(self, d_model, warmup_steps=4000):
                super(CustomSchedule, self).__init__()

                self.d_model = d_model
                self.d_model = tf.cast(self.d_model, tf.float32)

                self.warmup_steps = warmup_steps

            def __call__(self, step):
                step = tf.cast(step,tf.float32)
                arg1 = tf.math.rsqrt(step)
                arg2 = step * (self.warmup_steps ** -1.5)

                return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
        transformer = Transformer(num_layers= num_layers,
                            d_model=d_model,
                            num_heads=num_head,
                            dff=dff,
                            input_vocab_size=input_vocab_size,
                            target_vocab_size=target_vocab_size,
                            pe_input=pe_input,pe_target=pe_target)
        training = mode == tf.estimator.ModeKeys.TRAIN

        prediction, atte_weight = transformer(source,tar_inp,training, enc_padding_mask,
              combined_mask, dec_padding_mask)
        loss = loss_function(tar_real,prediction)
        learning_rate = CustomSchedule(d_model)(global_step)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate,beta1=0.9, beta2=0.98,
                                                     epsilon=1e-9)
        # new_global_step = global_step + 1

        # gradients = tape.gradient(loss,transformer.trainable_variables)
        # train_loss = tf.keras.metrics.Mean(name='train_loss')
        # train_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(
        #     name='train_accuracy')

        # optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
        #                                      epsilon=1e-9)
        train_accuracy = accuracy_function(tar_real, prediction)
        grads = optimizer.compute_gradients(loss)
        for grad, var in grads:
            tf.summary.histogram(var.name + '/gradient', grad,global_step)
        grad, var = zip(*grads)
        tf.clip_by_global_norm(grad, 0.5)
        grads = zip(grad, var)

        train_op = optimizer.apply_gradients(grads, global_step=global_step)
        # train_op = tf.group(train_op, [global_step.assign(new_global_step)])

        tf.summary.scalar('accuracy',train_accuracy,global_step)
        # = optimizer.minimize(lambda: loss,transformer.trainable_variables)
        # train_op = optimizer.apply_gradients(zip(gradients,transformer.trainable_variables))
        # train_loss(loss)
        # train_accuracy(tar_real, prediction)
        class TransformerRunHook(tf.estimator.SessionRunHook):
            def __init__(self):
                self.count = 0
                self.start_time  = time.time()
                self.ctime = time.time()
            def before_run(self, run_context):
                return tf.estimator.SessionRunArgs({'loss':loss,'accuracy':train_accuracy,'global_step':global_step,'learning_rate':learning_rate})

            def after_run(self, run_context, run_values):

                self.count += 1
                # a = np.mean(run_values.results['accuracy'])
                a = run_values.results['accuracy']
                if self.count % 1   == 0:
                    ntime = time.time()
                    dtime = ntime - self.ctime
                    self.ctime = ntime
                    print("Batch {0} : loss - {1:.3f} : accuracy - {2:.3f} : time_cost - {3:.2f} : all_time_cost - {4:.2f}".format(run_values.results['global_step'],run_values.results['loss'],a,dtime,ntime-self.start_time))
                pass
        # summaryHook = tf.estimator.SummarySaverHook(
        #     save_steps=10,
        #     output_dir='./transformer/summary',
        #     summary_writer=None,
        #     summary_op=tf.summary.
        # )

        return tf.estimator.EstimatorSpec(mode,{'target':prediction},loss,train_op,training_hooks=[TransformerRunHook()])

    return model_fn







class TransformerPredictor(Predictor):
    def __init__(self,topk = None):
        self.topk = topk
        model_fn = build_model_fn()
        self.estimator = tf.estimator.Estimator(model_fn, model_dir='./transformer', )


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




class Beamsearcher:

    def __init__(self,dataset,tokenizer,topk,predictor,max_count = 10):
        self.dataset = dataset
        self.topk = topk
        self.predictor = predictor
        self.buffer = [] # 保存当前生成内容
        self.gen_len = 0 # 保存生成长度
        self.next_topk = None
        self.tokenizer = tokenizer
        self.gen_result = []
        self.max_count = max_count


    def do_search(self,max_step):
        for case in self.dataset:
            source = case
            source_input = [source[:] for _ in range(max_step)]
            self.buffer = []

            self.buffer.append(([],0))
            for i in range(max_step):

                context = []
                # con_len = []

                for seq in self.buffer:
                    # con_len.append(len(seq[0]))
                    context.append(self.tokenizer.padding(seq[0][:],max_step))

                context = np.array(context)

                next_topk = self.predictor.predict(source_input[:context.shape[0]], context, self.gen_len,self.topk)
                tmp_buffer = []

                for i, val in enumerate(self.buffer):
                    candidate,score = val
                    for n in next_topk[i]:
                        ts = next_topk[i][n]
                        tc = candidate[:]
                        tc.append(n)
                        tmp_buffer.append((tc,score+ts))

                sorted(tmp_buffer,key=lambda x:x[1])
                tmp_buffer = tmp_buffer[:self.topk]
                self.buffer = tmp_buffer
                self.gen_len += 1


    def do_search_mt(self,max_step):
        # buffer_lock = threading.RLock()
        # result_lock = threading.RLock()
        buffer_lock = threading.BoundedSemaphore(1)
        result_lock = threading.BoundedSemaphore(self.topk)

        def fill_data(searcher):
            while len(searcher.gen_result) < searcher.max_count:

                # result_lock.acquire()
                if len(searcher.buffer) == 0 or searcher.gen_len == max_step:

                    source = next(searcher.dataset)
                    searcher.source_input = source[:]
                    searcher.gen_result.append(searcher.buffer)
                    searcher.buffer = []
                    for i in range(searcher.topk):
                        searcher.buffer.append(([],0))
                    searcher.gen_len = 0
                    searcher.next_topk = []
                else:
                    result_lock.acquire()
                    tmp_buffer = []
                    for i, val in enumerate(searcher.buffer):
                        candidate,score = val
                        for n in searcher.next_topk[i]:
                            ts = searcher.next_topk[i][n]
                            tc = candidate[:]
                            tc.append(n)
                            tmp_buffer.append((tc,score+ts))

                    sorted(tmp_buffer,key=lambda x:x[1])
                    tmp_buffer = tmp_buffer[:searcher.topk]
                    searcher.buffer = tmp_buffer
                    searcher.gen_len += 1
                    searcher.next_topk = []
                    result_lock.release()
                print('产生数据')

                context = []
                for seq in self.buffer:
                    # con_len.append(len(seq[0]))
                    context.append(self.tokenizer.padding(seq[0][:],max_step))
                searcher.context = context

                buffer_lock.release()

        def data_generator():
            searcher = self
            while len(searcher.gen_result) < searcher.max_count:
                buffer_lock.acquire()
                # buffer_lock.wait(2)
                for c in searcher.context:
                    print('消费数据')
                    yield {'source':tf.constant(searcher.source_input),'context':tf.constant(c)}, tf.constant(0)




        def input_fn():
            return tf.data.Dataset.from_generator(generator=data_generator,output_types=({'source':tf.int64,'context':tf.int64},tf.int64),output_shapes=({'source':[1000],'context':[100]},[])).batch(self.topk)

        model_fn = build_model_fn()
        self.estimator = tf.estimator.Estimator(model_fn, model_dir='./transformer', )
        pred = self.estimator.predict(input_fn,'target',yield_single_examples=False)

        def get_next(searcher):

            while len(searcher.gen_result) < searcher.max_count:
                # result_lock.acquire()
                res_v = []
                res = next(pred)
                print('生成下一步')
                for i,v in enumerate(res['target']):
                    vmap = v[searcher.gen_len]
                    sort_res = np.argsort(vmap)[-searcher.topk:]
                    map_res = {}
                    for k in sort_res:
                        map_res[k] = vmap[k]
                    res_v.append(map_res)
                searcher.next_topk = res_v
                # result_lock.notify()
                result_lock.release()
            # buffer_lock.release()

        producer = threading.Thread(target=fill_data,args=(self,))
        consumer = threading.Thread(target=get_next,args=(self,))
        producer.start()
        consumer.start()

        producer.join()
        consumer.join()







if __name__ == '__main__':


    # def generator():
    #     tokenizer = tokenization("/root/zsm/Summarization/news_data/NEWS_DICT.txt",DictSize=100000)
    #     source_file = queue_reader("NEWS","/home/user/zsm/Summarization/news_data")
    #     for line in source_file:
    #         example = line.split('#')
    #         title = example[0]
    #         desc = example[1]
    #         content = example[2]
    #         title = title.replace(' ','')
    #         content = content.replace(' ','')
    #         source_sequence = tokenizer.tokenize(content)
    #         source_sequence = tokenizer.padding(source_sequence,1000)
    #         title_sequence = tokenizer.tokenize(title)
    #         title_sequence = tokenizer.padding(title_sequence,100)
    #         # for i,s in enumerate(title_sequence):
    #         #     label = s
    #         #     context = title_sequence[:i]
    #         # #
    #         # feature = {
    #         #     'source':source_sequence,
    #         #     'context':title_sequence
    #         # }
    #         #     yield feature,label
    #         yield source_sequence,title_sequence
    #
    # g = generator()
    # for s,t in g:
    #     print(s)
    def train():
        model_fn = build_model_fn()
        estimator = tf.estimator.Estimator(model_fn,model_dir='./transformer',)
        input_fn = build_input_fn("NEWS", "/home/user/zsm/Summarization/news_data")

        estimator.train(input_fn,max_steps=1000000)


    def eval():
        model_fn = build_model_fn()
        estimator = tf.estimator.Estimator(model_fn, model_dir='./transformer', )
        input_fn = build_input_fn("E_NEWS", "/home/user/zsm/Summarization/news_data",batch_size=32)
        class EvalRunHook(tf.estimator.SessionRunHook):
            def __init__(self):
                self.count = 0
                self.start_time  = time.time()
                self.ctime = time.time()

            def after_run(self, run_context, run_values):

                self.count += 1
                # a = np.mean(run_values.results['accuracy'])
                if self.count % 1   == 0:
                    ntime = time.time()
                    dtime = ntime - self.ctime
                    self.ctime = ntime
                    print("Batch {0} : time_cost - {1:.2f} : all_time_cost - {2:.2f}".format(self.count, dtime, ntime- self.start_time))
                pass
        # estimator.train(input_fn,max_steps=1000000)
        # estimator.evaluate(input_fn, 1000,hooks=[EvalRunHook()])
        res = estimator.predict(input_fn,predict_keys=['target'])
        for i in res:

            print(i)
    #
    #
    def beamsearch():
        tokenizer = tokenization("/root/zsm/Summarization/news_data/NEWS_DICT.txt",DictSize=100000)
        source_file = queue_reader("E_NEWS", "/home/user/zsm/Summarization/news_data")
        predictor = TransformerPredictor(10)
        def _g():
            for source in source_file:
                source = ''.join(source.split('#')[0].split(' '))
                source = tokenizer.padding(tokenizer.tokenize(source),1000)
                yield  source
        g = _g()
        bs = Beamsearcher(dataset=g,tokenizer = tokenizer,topk=10,predictor=predictor)
        bs.do_search_mt(100)


    beamsearch()


