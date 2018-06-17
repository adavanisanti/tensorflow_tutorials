import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
from libs.utils import *
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

n_fc_dict = {0:32, 1:64, 2: 128, 3: 256}
print n_fc_dict
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

#n_fc = 2048
n_fc = n_fc_dict[rank]
W_fc = weight_variable([784,n_fc])
b_fc = bias_variable([n_fc])
h_fc = tf.nn.relu(tf.matmul(x,W_fc)+b_fc)

W_fc1 = weight_variable([n_fc,10])
b_fc1 = bias_variable([10])
y_pred = tf.nn.softmax(tf.matmul(h_fc,W_fc1)+b_fc1)

keep_prob = tf.placeholder(tf.float32)

cross_entropy = -tf.reduce_sum(y*tf.log(y_pred))
optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = mnist.train.num_examples
n_epochs = 10
for epoch_i in range(n_epochs):
    for batch_i in range(mnist.train.num_examples//batch_size):
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optimizer,feed_dict={x: batch_xs,y:batch_ys,keep_prob:0.5})
    print("Accuracy on rank %d: %f" %(rank,sess.run(accuracy,
                   feed_dict={
                       x: mnist.validation.images,
                       y: mnist.validation.labels,
                       keep_prob: 1.0
                   })))
