# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 18:50:20 2019

@author: Naveen_P08
"""

import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
sys.path.append('..')
import utils
import time

class CharRNNGRU:
    
    def __init__(self):
        tf.reset_default_graph()
        self.vocab = (" $%'()+,-./0123456789:;=?ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "\\^_abcdefghijklmnopqrstuvwxyz{|}")
        self.dataPath = 'data/arvix_abstracts.txt'
        self.hiddenSize = 200
        self.batchSize = 64
        self.numSteps = 50
        self.skipStep = 40
        self.temperatore = 0.7
        self.learningRate = 0.03
        self.lenGenerated = 300
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        
    def vocab_encode(self,text):
        return [self.vocab.index(x) + 1 for x in text if x in self.vocab]
    
    def vocab_decode(self,array):
        return ''.join(self.vocab[x-1] for x in array)
    
    def read_data(self):
        for text in open(self.dataPath) :
            encode = self.vocab_encode(text)
            for start in range(0,self.numSteps,self.numSteps//2):
                chunk = encode[start:start+self.numSteps]
                chunk += [0] * (self.numSteps - len(chunk))
                yield chunk
    def read_batch(self,stream):
        batch = []
        for encode in stream:
            batch.append(encode)
            if(len(batch) == self.batchSize):
                yield batch
                batch = []
        #yield batch
            
    def create_one_hot(self):
        self.seq = tf.placeholder(tf.int32,[None,None])
        self.one_hot = tf.one_hot(self.seq,len(self.vocab))
        self.seq_sign = tf.sign(self.one_hot)
        self.max_value = tf.reduce_max(self.seq_sign, 2)
        self.length = tf.reduce_sum(self.max_value, 1)
        
    def create_rnn(self):
        self.rnn_cell = tf.contrib.rnn.GRUCell(self.hiddenSize)
        self.in_state = tf.placeholder_with_default(self.rnn_cell.zero_state(tf.shape(self.one_hot)[0],tf.float32),[None,self.hiddenSize])
        self.output,self.out_state = tf.nn.dynamic_rnn(self.rnn_cell,self.one_hot,self.length,self.in_state)
        
    def create_model(self):
        self.create_one_hot()
        self.create_rnn()
        self.logits = tf.contrib.layers.fully_connected(self.output,len(self.vocab),None)
        self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits[:,:-1],labels=self.one_hot[:,1:]))
        self.optimizer = tf.train.AdamOptimizer(self.learningRate).minimize(self.loss,global_step=self.global_step)
        
    def create_sample(self):
        self.probs = tf.exp(self.logits[:,-1]/self.temperatore)
        self.sample = tf.multinomial(self.probs,1)[:,0]
        
    def training(self):
        self.create_model()
        self.create_sample()
        saver = tf.train.Saver()
        start = time.time()
        utils.make_dir('checkpoints2')
        with tf.Session() as sess:
            writer = tf.summary.FileWriter('graphs/gist', sess.graph)
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints2/rnn_naveen/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                
            iteration = self.global_step.eval()
            
            for batch in self.read_batch(self.read_data()):
                if(len(batch) <2 ) :
                    print("Batch shape {0} \n {1} \n {2}".format(tf.shape(batch),self.vocab_decode(batch[0]),self.vocab_decode(batch[1])))
                batch_loss,_ = sess.run([self.loss,self.optimizer],feed_dict={self.seq:batch})
                if (iteration+1)%self.skipStep == 0 :
                    print("Iteration : {0} Batch Loss : {1} Time taken : {2} logits : {3}".
                          format(iteration,batch_loss,time.time()-start,self.logits[0][0][0]))
                    self.sampling_inference(sess)
                    start = time.time()
                    saver.save(sess, 'checkpoints2/rnn_naveen/char-rnn', iteration)
                iteration +=1
                
    def sampling_inference(self,sess):
        sentence = "T"
        state = None
        for _ in range (self.lenGenerated):
            batch = [self.vocab_encode(sentence[-1])]
            feed = {self.seq:batch}
            
            if state is not None:
                feed.update({self.in_state:state})
                
            index,state = sess.run([self.sample,self.out_state],feed)
            sentence += self.vocab_decode(index)
        print("Sampling text--------------\n {0} \n-------------------".format(sentence))
        
    def test_one_hot(self):
        self.create_model()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            batch = next(self.read_batch(self.read_data()))
            print("Text {0}".format(self.vocab_decode(batch[0])))
            print("Encode {0}".format([text for text in batch[0]]))
            print("seq {0}".format(sess.run(self.seq,feed_dict={self.seq:batch})))
            print("One hot {0}".format(sess.run(self.one_hot,feed_dict={self.seq:batch})))
            print("Seq sign {0}".format(sess.run(self.seq_sign,feed_dict={self.seq:batch})))
            print("Max len {0} \n {1}".format(sess.run(self.max_value,feed_dict={self.seq:batch}),self.max_value.get_shape()))
            print("len {0} \n size of len {1}".format(sess.run(self.length,feed_dict={self.seq:batch}),self.length.get_shape()))
            print("Logits {0}".format(sess.run(tf.reduce_sum(self.logits,0),feed_dict={self.seq:batch})))
            print("Logits {0}".format(sess.run(tf.reduce_sum(self.logits,1),feed_dict={self.seq:batch})))
            print("Logits {0}".format(sess.run(tf.reduce_sum(self.logits,2),feed_dict={self.seq:batch})))
            self.create_sample()
            print("Probs {0} \n {1}".format(sess.run(self.probs,feed_dict={self.seq:batch}),self.probs.shape))
            index = sess.run(self.sample,feed_dict={self.seq:batch})
            print("Sample {0} len {1}".format(index,len(index)))
            print("Decode {0}".format(self.vocab_decode(index)))
        
    def test_encoding(self):
        chunk = self.read_data()
        for _ in range(10):
            encode = next(chunk)
            text = self.vocab_decode(encode)
            print(encode)
            print(text)
        
def main():
    rnn = CharRNNGRU()
    rnn.training()
    #rnn.test_one_hot()
    
        
    

if __name__ == '__main__':
    main()
    