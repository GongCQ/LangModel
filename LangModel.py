import tensorflow as tf
import numpy as np
import datetime as dt

class LangModel:
    def __init__(self, vocabSize, isTrain, batchSize, numSteps,
                 hiddenSize = 20, numLstmLayers = 2,
                 keepProb = 0.5, maxGradNorm = 5, learnRate = 0.1):
        self.hiddenSize = hiddenSize
        self.batchSize = batchSize
        self.numSteps = numSteps
        self.vocabSize = vocabSize
        self.keepProb = keepProb
        self.numLstmLayers = numLstmLayers

        self.sess = tf.InteractiveSession()
        self.graph = tf.Graph()
        self.graph.as_default()

        # 0.input layer and objective value
        self.input = tf.placeholder(tf.int32, [self.batchSize, self.numSteps], name='input')
        self.objVal = tf.placeholder(tf.float32, [self.batchSize, self.numSteps, self.vocabSize], name='objVal')

        # 1.word embedding layer
        emb = tf.get_variable(name='emb', shape=[self.vocabSize, self.hiddenSize])
        wordEmb = tf.nn.embedding_lookup(emb, self.input, name='wordEmb')
        if isTrain:
            wordEmb = tf.nn.dropout(wordEmb, self.keepProb, name='wordEmbWithDropout')

        # 2.recurrent network with lstm cell, and initial state
        lstmCell = tf.nn.rnn_cell.BasicLSTMCell(self.hiddenSize)
        if isTrain:
            lstmCell = tf.nn.rnn_cell.DropoutWrapper(lstmCell, output_keep_prob=self.keepProb)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstmCell] * self.numLstmLayers)
        self.initState = cell.zero_state(self.batchSize, tf.float32)

        # 3.output layer
        state = self.initState
        outputSteps = []
        with tf.variable_scope('recurrent') as recurrentScope:
            for step in range(self.numSteps):
                if step > 0:
                    recurrentScope.reuse_variables()
                output, state = cell(wordEmb[:, step, :], state)
                outputSteps.append(output)
        outputSeq = tf.concat(outputSteps, 0) # outputSeq = tf.reshape(tf.concat(outputSteps, 1), shape=[-1, self.hiddenSize])
        self.finalState = state

        # 4.word unembedding, a full connection layer
        wordUnembWeight = tf.get_variable('wordUnembWeight', initializer=tf.truncated_normal([self.hiddenSize, self.vocabSize], stddev=0.1))
        wordUnembBias = tf.get_variable('wordUnembBias', initializer=tf.zeros([self.vocabSize]))
        self.output = tf.matmul(outputSeq, wordUnembWeight) + wordUnembBias

        # 5.loss function
        lossWeight = tf.ones(shape=[self.batchSize * self.numSteps], dtype=tf.float32, name='lossWeight')
        objValReshape = tf.reshape(tf.transpose(self.objVal, perm=[1, 0, 2]), [self.batchSize * self.numSteps, self.vocabSize])
        loss = tf.losses.softmax_cross_entropy(objValReshape, self.output, weights=lossWeight, reduction=tf.losses.Reduction.NONE)
        self.cost = tf.reduce_mean(loss, axis=0) #

        # train and optimize
        if not isTrain:
            return
        trainVar = tf.trainable_variables()
        grad, norm = tf.clip_by_global_norm(tf.gradients(self.cost, trainVar), maxGradNorm)
        optimizer = tf.train.GradientDescentOptimizer(learnRate)
        self.trainOper = optimizer.apply_gradients(zip(grad, trainVar)) # PS: If "tf.clip_by_global_norm" is not necessary,
                                                                        #     these statements can be written as "minimize" briefly.
                                                                        #     See function "minimize", "compute_gradients", "apply_gradients"
                                                                        #     in class "tf.train.GradientDescentOptimizer" in API document
    def GetInitState(self):
        return self.sess.run(self.initState)

def Train(batchs, vocabSize, batchSize, numSteps):
    model = LangModel(vocabSize, True, batchSize, numSteps)
    model.sess.run(tf.global_variables_initializer())
    initState = model.GetInitState()
    totalCost = 0
    sess = tf.Session()
    for b in range(len(batchs)):
        batch = batchs[b]
        input = batch[0]
        objVal = tf.one_hot(batch[1], vocabSize)
        cost, finalState, _ = model.sess.run([model.cost, model.finalState, model.trainOper],
                                              feed_dict={model.input: input,
                                                         model.objVal: objVal.eval(session=sess),
                                                         model.initState: initState})
        totalCost += cost
        print(str(dt.datetime.now()))
        print('train batchSeq=' + str(b) + '/' + str(len(batchs)) + ', perplex=' + str(np.exp(cost)) + ', cost=' + str(cost))

    print(np.exp(totalCost / len(batchs)))
    return np.exp(totalCost / len(batchs))

def Test(batchs, vocabSize, batchSize, numSteps):
    model = LangModel(vocabSize, False, batchSize, numSteps)
    # model.sess.run(tf.global_variables_initializer())
    initState = model.GetInitState()
    totalCost = 0
    sess = tf.Session()
    for b in range(len(batchs)):
        batch = batchs[b]
        input = batch[0]
        objVal = tf.one_hot(batch[1], vocabSize)
        cost = model.sess.run(model.cost, feed_dict={model.input: input,
                                                     model.objVal: objVal.eval(session=sess),
                                                     model.initState: initState})
        totalCost += cost
        print('test batchSeq=' + str(b) + '/' + str(len(batchs)) + ', perplex=' + str(np.exp(cost)) + ', cost=' + str(cost))
    print(np.exp(totalCost / len(batchs)))
    return np.exp(totalCost / len(batchs))


from sample import batchs, vocabSize, batchSize, numSteps
print('success to get batchs')

divide = int(len(batchs) * 0.7)
trainBatch = batchs[0 : divide]
testBatch = batchs[divide : ]

print('begin to train')
Train(trainBatch, vocabSize, batchSize=batchSize, numSteps=numSteps)
print('begin to test')
Test(testBatch, vocabSize, batchSize=batchSize, numSteps=numSteps)