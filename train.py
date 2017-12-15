
from __future__ import division
from __future__ import print_function

import glob
import json
import math
import numpy as np
import os
import random
import sys
import tensorflow as tf
import time

import audio_producer
import models

tf.app.flags.DEFINE_string("config", "configs/kws.json",
    "Configuration json for model building and training")
tf.app.flags.DEFINE_integer("num_gpus", 1,
    "Number of GPUs to train with.")
FLAGS = tf.app.flags.FLAGS

def check_path(path):
    if os.path.exists(path):
        overwrite = raw_input(("The path \'{}\' exists. Do you want "
                               "to use it anyway (y)? ").format(path))
        if overwrite != 'y':
            sys.exit(0)
    else:
        os.mkdir(path)

def run_epoch(model, producer, session, save_path, saver):

    summary_writer = tf.summary.FileWriter(save_path, flush_secs=30)
    model_path = os.path.join(save_path, "model.ckpt")
    summary_op = tf.summary.scalar('cost', model.avg_cost)

    ops = [model.grad_norm, model.cost, model.avg_cost,
           model.global_step, summary_op, model.train_op]

    start_time = time.time()
    compute_time = 0

    step, = session.run([model.global_step])
    sort = (step == 0)

    for e, (inputs, labels) in enumerate(producer.iterator(sort=sort)):

        compute_time -= time.time()
        feed_dict = model.feed_dict(inputs, labels)
        res = session.run(ops, feed_dict)
        grad_norm, cost, avg_cost, step, summary, _ = res
        compute_time += time.time()
        if math.isnan(grad_norm):
            print("NaN GradNorm. Exiting")
            import sys
            sys.exit(1)

        if step == 100:
            model.start_momentum(session)

        if step % 1000 == 0:
            saver.save(session, model_path)
        summary_writer.add_summary(summary, global_step=step)

        log_str = ("Iter {}: AvgCost {}, Cost {:.2f}, "
                   "GradNorm {:.2f}, CumTime {:.2f} (s), "
                   "CompTime {:.2f} (s), AvgItTime {:.2f} (s)")
        cum_time = time.time() - start_time
        print(log_str.format(step, avg_cost, cost,
                grad_norm, cum_time,
                compute_time, cum_time / (e + 1)))

    saver.save(session, model_path)
    print("Total time: ", time.time() - start_time)

def main(argv=None):
    with open(FLAGS.config) as fid:
        config = json.load(fid)

    train_jsons = config['data']['train_jsons']
    sample_rate = config['data']['sample_rate']
    batch_size = config['inference']['batch_size']
    epochs = config['train']['epochs']

    # TODO, awni, for now it's on the user to get this right.
    # E.g. all the config params which have to remain the same
    # to "restore" a model.
    restore_path = config['io'].get('restore_path', None)

    save_path = config['io']['save_path']
    check_path(save_path)

    producer = audio_producer.AudioProducer(train_jsons, batch_size,
                                            sample_rate=sample_rate)
    config['inference']['alphabet_size'] = producer.alphabet_size
    config['inference']['input_dim'] = producer.input_dim

    with open(os.path.join(save_path, "model.json"), 'w') as fid:
        json.dump(config, fid)

    with tf.Graph().as_default():

        model = models.MultiSpeechModel(FLAGS.num_gpus)
        model.init_inference(config['inference'])
        model.init_cost()
        model.init_train(config['train'])

        sess_conf = tf.ConfigProto(allow_soft_placement=True)
        session = tf.Session(config=sess_conf)
        saver = tf.train.Saver()
        if restore_path:
            saver.restore(session, os.path.join(restore_path, "model.ckpt"))
        else:
            session.run(tf.global_variables_initializer())
            print("Estimating and setting the mean and standard...")
            mean, std = producer.estimate_mean_std()
            model.set_mean_std(mean, std, session)

        print("Begin training...")
        for e in range(epochs):
            run_epoch(model, producer, session, save_path, saver)
            print("Finished epoch", e)

if __name__=="__main__":

    # For determinism
    random.seed(10)
    np.random.seed(10)

    tf.app.run()

