import tensorflow as tf

with tf.variable_scope('vs') as vs:
    gv0 = tf.get_variable('gv0', shape=[1,2], dtype=tf.float64)
    V0 = tf.Variable(tf.zeros([1], dtype=tf.float64), dtype=tf.float64, name='V0')
    print(vs.name)
    print(gv0.name)
    print(V0.name)
    print(tf.get_variable_scope().name)
print('----------')
with tf.name_scope('ns') as ns:
    gv0 = tf.get_variable('gv0', shape=[1,2], dtype=tf.float64)
    V0 = tf.Variable(tf.zeros([1], dtype=tf.float64), dtype=tf.float64, name='V0')
    print(ns)
    print(gv0.name)
    print(V0.name)
    print(tf.get_variable_scope().name)

ph = tf.placeholder(tf.int32, [1,2,3], name='ph')
ph0 = tf.reshape(ph, [-1])
ph1 = tf.reshape(ph, [1])
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
e = ph1.eval(feed_dict={ph: [[[2,3,4], [5,6,7]]]})
ddd = 0