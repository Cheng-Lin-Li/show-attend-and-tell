'''
Reference: https://github.com/coldmanck/show-attend-and-tell

This is a distributed parallel computing version on Clusterone environment.
'''

#!/usr/bin/python
import tensorflow as tf

from config import Config
from model import CaptionGenerator
from dataset import prepare_train_data, prepare_eval_data, prepare_test_data

from clusterone_config import distributed_env

tf.flags.DEFINE_string('phase', 'train',
                       'The phase can be train, eval or test')

tf.flags.DEFINE_boolean('load', False,
                        'Turn on to load a pretrained model from either \
                        the latest checkpoint or a specified file')

tf.flags.DEFINE_string('model_file', None,
                       'If sepcified, load a pretrained model from this file')

tf.flags.DEFINE_boolean('load_cnn', False,
                        'Turn on to load a pretrained CNN model')

tf.flags.DEFINE_string('cnn_model_file', './vgg16_no_fc.npy',
                       'The file containing a pretrained CNN model')

tf.flags.DEFINE_boolean('train_cnn', False,
                        'Turn on to train both CNN and RNN. \
                         Otherwise, only RNN is trained')

tf.flags.DEFINE_integer('beam_size', 3,
                        'The size of beam search for caption generation')

def main(argv):
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    config = Config()
    config.phase = FLAGS.phase
    config.train_cnn = FLAGS.train_cnn
    config.beam_size = FLAGS.beam_size

    # Cluster One setting
    clusterone_dist_env = distributed_env(config.root_path_to_local_data,
                                          config.path_to_local_logs,
                                          config.cloud_path_to_data,
                                          config.local_repo,
                                          config.cloud_user_repo,
                                          flags)

    clusterone_dist_env.get_env()

    tf.reset_default_graph()
    device, target = clusterone_dist_env.device_and_target() # getting node environment
    # end of setting

# Using tensorflow's MonitoredTrainingSession to take care of checkpoints
    with tf.train.MonitoredTrainingSession(
        master=target,
        is_chief=(FLAGS.task_index == 0),
        checkpoint_dir=FLAGS.log_dir) as sess:

#     with tf.Session() as sess:
        if FLAGS.phase == 'train':
            # training phase
            data = prepare_train_data(config)
            with tf.device(device): # define model
                model = CaptionGenerator(config)
            sess.run(tf.global_variables_initializer())
            if FLAGS.load:
                model.load(sess, FLAGS.model_file)
            if FLAGS.load_cnn:
                model.load_cnn(sess, FLAGS.cnn_model_file)
            tf.get_default_graph().finalize()
            model.train(sess, data)

        elif FLAGS.phase == 'eval':
            # evaluation phase
            config.batch_size = 1
            coco, data, vocabulary = prepare_eval_data(config)
            with tf.device(device): # define model
                model = CaptionGenerator(config)
                model.load(sess, FLAGS.model_file)
                tf.get_default_graph().finalize()
            model.eval(sess, coco, data, vocabulary)

        else:
            # testing phase
            data, vocabulary = prepare_test_data(config)
            with tf.device(device): # define model
                model = CaptionGenerator(config)
                model.load(sess, FLAGS.model_file)
                tf.get_default_graph().finalize()
            model.test(sess, data, vocabulary)

if __name__ == '__main__':
    tf.app.run()
