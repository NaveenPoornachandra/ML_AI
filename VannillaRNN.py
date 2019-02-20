# Imports 

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import utils

class VanillaRNNMNIST :
    def __init__(self) :
        tf.reset_default_graph()
        """
        Constants :
        """
        self.num_inputs = 28 # Dataset 28 X 28
        self.timesteps = 28 # 28 timesteps
        self.n_classes = 10 # number of classes. one class per digit
        
        """
        Hyperparameters :
        """
        self.learning_rate = 0.001 # The optimization initial learning rate
        self.epochs = 10           # Total number of training epochs
        self.batch_size = 100      # Training batch size
        self.display_freq = 100    # Frequency of displaying the training results
        
        self.global_step = 0
        
        """
        Network parameter :
        """
        self.num_hidden_units = 128  # Number of hidden units of the RNN
        
    def load_data(self,mode='train'):
        """
        Function to download MNIST dataset
        :param_mode: train or test
        :return:image and corresponding labels
        
        """
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
        if mode == 'train':
            x_train,y_train,x_valid,y_valid = mnist.train.images,mnist.train.labels,\
                                              mnist.validation.images,mnist.validation.labels
            return x_train,y_train,x_valid,y_valid
        elif mode == 'test':
            x_test,y_test = mnist.test.images,mnist.test.labels
            return x_test,y_test
        
    def randomize(self,x,y):
        """
        Randomize the order of samples and corresponding labels 
        """
        permutation = np.random.permutation(y.shape[0])
        shuffled_x = x[permutation,:]
        shuffled_y = y[permutation]
        return shuffled_x,shuffled_y
    
    def get_next_batch(self,x,y,start,end):
        x_batch = x[start:end]
        y_batch = y[start:end]
        return x_batch,y_batch
    
    def bias(self,shape):
        """
        Create a weight variable with appropriate initialization
        """
        return tf.Variable(tf.zeros(shape),name='bias',dtype=tf.float32)
        
    def weights(self,shape):
        """
        Create a weight variable with appropriate initialization
        """
        return tf.Variable(tf.truncated_normal(shape, stddev=0.01),dtype=tf.float32,name='weight')
        
    def rnn(self):
        x_input = tf.unstack(self.x,self.timesteps,1)
        rnn_cell = rnn.BasicRNNCell(self.num_hidden_units)
        states_series,current_state = rnn.static_rnn(rnn_cell,x_input,dtype=tf.float32)
        return tf.matmul(current_state,self.w)+self.b
    
    def create_network(self):
        self.x = tf.placeholder(tf.float32,shape=[None,self.timesteps,self.num_inputs],name='x')
        self.y = tf.placeholder(tf.float32,shape=[None,self.n_classes],name='y')
        self.w = self.weights(shape=[self.num_hidden_units,self.n_classes])
        self.b = self.bias(shape=[self.n_classes])
        self.logits = self.rnn()
        self.y_pred = tf.nn.softmax(self.logits)
        
        # Model prediction
        self.class_prediction = tf.argmax(self.logits,axis=1,name='predictions')
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y,logits=self.logits),name='loss')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,name='Adam-op').minimize(self.loss)
        self.correct_predictions = tf.equal(tf.argmax(self.logits,1),tf.argmax(self.y,1),name='correct_pred')
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions,tf.float32),name='accuracy')
        
    def train(self):
        saver = tf.train.Saver()
        utils.make_dir('vanilla_rnn_chkpts')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            chkpt = tf.train.get_checkpoint_state(os.path.dirname('vanilla_rnn_chkpts/checkpoint'))
            
            if chkpt and chkpt.model_checkpoint_path :
                saver.restore(sess,chkpt.model_checkpoint_path)
                
            x_train,y_train,x_valid,y_valid = self.load_data()
            num_tr_iter = int(len(y_train) / self.batch_size)
            for epoch in range(self.epochs):
                print('Training epoch: {}'.format(epoch + 1))
                x_train, y_train = self.randomize(x_train, y_train)
                for iteration in range(self.global_step,num_tr_iter):
                    self.global_step += 1
                    start = iteration * self.batch_size
                    end = (iteration + 1) * self.batch_size
                    x_batch, y_batch = self.get_next_batch(x_train, y_train, start, end)
                    x_batch = x_batch.reshape((self.batch_size, self.timesteps, self.num_inputs))
                    # Run optimization op (backprop)
                    feed_dict_batch = {self.x: x_batch, self.y: y_batch}
                    sess.run(self.optimizer, feed_dict=feed_dict_batch)
            
                    if (iteration+1) % self.display_freq == 0:
                        # Calculate and display the batch loss and accuracy
                        loss_batch, acc_batch = sess.run([self.loss, self.accuracy],
                                                         feed_dict=feed_dict_batch)
                        
                        saver.save(sess,'vanilla_rnn_chkpts/checkpoint',iteration)
                        
                        print("iter {0:3d}:\t Loss={1:.2f},\tTraining Accuracy={2:.01%}".
                              format(iteration, loss_batch, acc_batch))
            
                # Run validation after every epoch
            
                feed_dict_valid = {self.x: x_valid[:1000].reshape((-1, self.timesteps, self.num_inputs)), self.y: y_valid[:1000]}
                loss_valid, acc_valid = sess.run([self.loss, self.accuracy], feed_dict=feed_dict_valid)
                print('---------------------------------------------------------')
                print("Epoch: {0}, validation loss: {1:.2f}, validation accuracy: {2:.01%}".
                      format(epoch + 1, loss_valid, acc_valid))
                print('---------------------------------------------------------')
                
    def plot_images(self,images,cls_true,cls_pred=None,title=None):
        """
        Create figure with 3 X 3 subplots.
        :param images: array of images to be ploted, (9,image_height * image_width)
        :param cls_true : corresponding true lables (9,)
        :param cls_pred : Corresponding predicted labels (9,)
        
        """
        fig,axes = plt.subplots(3,3,figsize=(9,9))
        fig.subplots_adjust(hspace=0.3,wspace=0.3)
        for i, ax in enumerate(axes.flat):
            # Plot image
            ax.imshow(np.squeeze(images[i]).reshape(28,28),cmap='binary')
            
            # Show true and prediction classes
            if cls_pred is None:
                ax_title = "True {0}".format(cls_true[i])
            else:
                ax_title = "True {0}, Pred {1}".format(cls_true[i],cls_pred[i])
                
            ax.set_title(ax_title)
            
            # Remove ticks from the plot
            ax.set_xticks([])
            ax.set_yticks([])
            
            if title :
                plt.suptitle(title,size=20)
        plt.show(block=False)
            
    def plot_example_errors(self,images,cls_true,cls_pred,title=None):
        """
        Function for plotting examples that have been miss classified
        :param-images : array of all images
        :param cls_true
        :param cls_pred
        
        """
        
        incorrect = np.logical_not(np.equal(cls_pred,cls_true))
        
        # Get the images from the test class that have been incorrectly classified
        
        incorrect_images = images[incorrect]
        
        # Get the true and predicted classes of those classes
        
        cls_pred = cls_pred[incorrect]
        cls_true = cls_true[incorrect]
        
        self.plot_images(images=incorrect_images[0:9],cls_true=cls_true[0:9],cls_pred=cls_pred[0:9],title=title)
    
    def run_test(self):
        saver = tf.train.Saver()
        x_test, y_test  = self.load_data(mode='test')
        
        feed_dict_test = {self.x:x_test[:1000].reshape((-1,self.timesteps,self.num_inputs)),self.y:y_test[:1000]}
        
        with tf.Session() as sess :
            sess.run(tf.global_variables_initializer())
            chkpt = tf.train.get_checkpoint_state(os.path.dirname('vanilla_rnn_chkpts/checkpoint'))
            
            if chkpt and chkpt.model_checkpoint_path :
                saver.restore(sess,chkpt.model_checkpoint_path)
                
            loss_test,acc_test = sess.run([self.loss, self.accuracy],feed_dict=feed_dict_test)
            print('---------------------------------------------------------')
            print("Test loss: {0:.2f}, test accuracy: {1:.01%}".format(loss_test, acc_test))
            print('---------------------------------------------------------')
            
            # Plot correct and misclassified
            cls_pred = sess.run(self.class_prediction,feed_dict=feed_dict_test)
            cls_true = np.argmax(y_test, axis=1)
            self.plot_images(x_test, cls_true, cls_pred, title='Correct Examples')
            self.plot_example_errors(x_test[:1000], cls_true[:1000], cls_pred, title='Misclassified Examples')
            plt.show()
            
            
            
        
def main():
    model = VanillaRNNMNIST()
    model.load_data()
    model.create_network()
    model.train()
    model.run_test()
    
if __name__ == '__main__' :
    main()
            
                                        