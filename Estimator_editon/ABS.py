
import sys
sys.path.append('/home/user/zsm/Summarization')
import tensorflow as tf
import numpy as np
from data_util.tokenization import tokenization
from util.file_utils import queue_reader
from model.ABS import ABS
import time


def build_input_fn(name,data_set,batch_size = 32,context_len = 10):

    def generator():
        tokenizer = tokenization("/root/zsm/Summarization/news_data/NEWS_DICT.txt",DictSize=100000)
        source_file = queue_reader(name,data_set)
        for line in source_file:
            try:
                example = line.split('#')
                title = example[0]
                desc = example[1]
                content = example[2]
                title = title.split(' ')
                content = content.split(' ')
                source_sequence = tokenizer.tokenize(content)
                source_sequence = tokenizer.padding(source_sequence,1000)
                title_sequence = tokenizer.tokenize(title)
                title_sequence = tokenizer.padding(title_sequence,100)
                title_context = []
                for i,s in enumerate(title_sequence):
                    if i > context_len:
                        context = title_sequence[i-1-context_len:i-1]
                    else:
                        context = [0]*(context_len-1-i)
                        context = context.extend(title_sequence[:i-1])
                    title_context.append(context)
                feature = {
                    'source':source_sequence,
                    'context':title_context,
                    'title':title_sequence
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





def build_model_fn(lr = 0.01,context_len = 10,d_model=200,input_vocab_size=100000,hidden_size=200):

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
        title = features['title']
        title_real = title[:,1:]
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
        training = mode == tf.estimator.ModeKeys.TRAIN

        ABS_model = ABS(d_model=d_model,context_len=context_len,input_vocab_size=input_vocab_size,hidden_size=hidden_size)

        prediction = ABS_model(source,context_len,training)
        loss = loss_function(title_real,prediction)
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
        train_accuracy = accuracy_function(title_real, prediction)
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
                if self.count % 10  == 0:
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