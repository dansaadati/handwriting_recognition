import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt




def run_model(session, predict, loss_val, Xd, yd,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False):
    # have tensorflow compute accuracy
    correct_prediction = tf.equal(tf.argmax(predict,1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)
    
    training_now = training is not None
    
    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [mean_loss,correct_prediction,accuracy]
    if training_now:
        variables[-1] = training
    
    # counter 
    iter_cnt = 0
    for e in range(epochs):
        # keep track of losses and accuracy
        correct = 0
        losses = []
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
            # generate indicies for the batch
            start_idx = (i*batch_size)%X_train.shape[0]
            idx = train_indicies[start_idx:start_idx+batch_size]
            
            # create a feed dictionary for this batch
            feed_dict = {X: Xd[idx,:],
                         y: yd[idx],
                         is_training: training_now }
            # get batch size
            actual_batch_size = yd[i:i+batch_size].shape[0]
            
            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss, corr, _ = session.run(variables,feed_dict=feed_dict)
            
            # aggregate performance stats
            losses.append(loss*actual_batch_size)
            correct += np.sum(corr)
            
            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
                      .format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
            iter_cnt += 1
        total_correct = correct/Xd.shape[0]
        total_loss = np.sum(losses)/Xd.shape[0]
        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"\
              .format(total_loss,total_correct,e+1))
        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e+1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.show()
    return total_loss,total_correct


with tf.Session() as sess:
    with tf.device("/cpu:0"): #"/cpu:0" or "/gpu:0" 
        sess.run(tf.global_variables_initializer())
        print('Training')
        run_model(sess,y_out,mean_loss,X_train,y_train,1,64,100,train_step,True)
        print('Validation')
        run_model(sess,y_out,mean_loss,X_val,y_val,1,64)




def my_model(X,y,is_training):
     
    Wconv1 = tf.get_variable("Wconv1", shape=[3, 3, 3, 32])	
    bconv1 = tf.get_variable("bconv1", shape=[32])
    Wconv2 = tf.get_variable("Wconv2", shape=[5, 5, 32, 32])
    bconv2 = tf.get_variable("bconv2", shape=[32])
    hidden_dim1 = 1024
    W1 = tf.get_variable("W1", shape=[3872, hidden_dim1])
    b1 = tf.get_variable("b1", shape=[hidden_dim1])
    W2 = tf.get_variable("W2", shape=[1024, 10])
    b2 = tf.get_variable("b2", shape=[10])
    
    # define our graph (e.g. two_layer_convnet)
    a1 = tf.nn.conv2d(X, Wconv1, strides=[1,1,1,1], padding='VALID') + bconv1
    # batchnorm bn1
    bn1 = tf.contrib.layers.batch_norm(a1, center=True)
    h1 = tf.nn.relu(bn1)

    p1 = tf.nn.max_pool(h1, [1, 2, 2, 1], [1, 2, 2, 1], padding = 'VALID')
    #
    t1 = tf.nn.conv2d(p1, Wconv2, strides=[1,1,1,1], padding='VALID') + bconv2
    t3 = tf.contrib.layers.batch_norm(t1, center=True)
    t2 = tf.nn.relu(t3)   
    #
    #p1_flat = tf.reshape(p1,[-1,7200])
    p1_flat = tf.reshape(t2,[-1, 3872])
    h2 = tf.matmul(p1_flat,W1) + b1
    h3 = tf.nn.relu(h2)
    y_out = tf.matmul(h3,W2) + b2
    return y_out


total_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y,10),logits=y_out)
mean_loss = tf.reduce_mean(total_loss)

optimizer = tf.train.RMSPropOptimizer(1e-3)

