import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
from libs.utils import *
import horovod.tensorflow as hvd

hvd.init()
config = tf.ConfigProto()
config.gpu_options.visible_device_list = str(hvd.local_rank())

mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

n_fc = 32
W_fc = weight_variable([784,n_fc])
b_fc = bias_variable([n_fc])
h_fc = tf.nn.relu(tf.matmul(x,W_fc)+b_fc)

W_fc1 = weight_variable([n_fc,10])
b_fc1 = bias_variable([10])
y_pred = tf.nn.softmax(tf.matmul(h_fc,W_fc1)+b_fc1)

keep_prob = tf.placeholder(tf.float32)

cross_entropy = -tf.reduce_sum(y*tf.log(y_pred))
optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)

optimizer = hvd.DistributedOptimizer(optimizer)
hooks = [hvd.BroadcastGlobalVariablesHook(0)]

# Make training operation
train_op = optimizer.minimize(cross_entropy)

# Save checkpoints only on worker 0 to prevent other workers from corrupting them.
checkpoint_dir = '/tmp/train_logs' if hvd.rank() == 0 else None

# The MonitoredTrainingSession takes care of session initialization,
# restoring from a checkpoint, saving to a checkpoint, and closing when done
# or an error occurs.
with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                       config=config,
                                       hooks=hooks) as mon_sess:
    while not mon_sess.should_stop():
        mon_sess.run(train_op)
