import numpy as np
import pandas as pd
import torch

class DataLoad(object):
    def __init__(self,train_data_file,test_data_file,delimiter,number_of_features,batch_size,debug):
        """ Loading training data """
        self.training_set = pd.read_csv(train_data_file,delimiter = delimiter)
        self.training_set = np.array(self.training_set,dtype='int')
        """ Loading testing data """
        self.test_set = pd.read_csv(test_data_file,delimiter = delimiter)
        self.test_set = np.array(self.test_set,dtype='int')
        """ Determining number of users and number of movies """
        self.number_of_users = int(max(max(self.training_set[:,0]),max(self.test_set[:,0])))
        self.number_of_movies = int(max(max(self.training_set[:,1]),max(self.test_set[:,1])))
        """ Setting number of hidden nodes """
        self.number_of_hidden_nodes = number_of_features
        self.batch_size = batch_size
        
        if(debug):
            print("---------------Input Data begin----------------------\n")
            print(" Numner of users = {0}\n Number of movies = {1} \n Number of hidden nodes = {2} \n"
                  .format(self.number_of_users,self.number_of_movies,self.number_of_hidden_nodes))
            print("---------------Input Data end----------------------\n")
            
        self.training_set = self.convert(self.training_set)
        self.test_set = self.convert(self.test_set)
        
        if(debug):
            print("---------------Training and Testing data begin----------------------\n")
            #print(" Training set = {0}\n Testing set = {1} \n"
            #      .format(self.training_set,self.test_set))
            print("---------------Training and Testing data end----------------------\n")
        
        self.training_set = torch.FloatTensor(self.training_set)
        self.test_set = torch.FloatTensor(self.test_set)
        
    def convert(self,data):
        converted_data = []
        for id_users in range(1,self.number_of_users+1):
            ratings = np.zeros(self.number_of_movies)
            id_movies = data[:,1][data[:,0] == id_users]
            id_ratings = data[:,2][data[:,0] == id_users]
            ratings[id_movies - 1] = id_ratings
            converted_data.append(list(ratings))
        return converted_data
        

class RestrictedBoltzmannMachine(DataLoad):
    def __init__(self,train_data_file,test_data_file,delimiter,number_of_features,batch_size,no_epochs,debug=False):
        DataLoad.__init__(self,train_data_file,test_data_file,delimiter,number_of_features,batch_size,debug)
        self.weights = torch.randn(self.number_of_hidden_nodes,self.number_of_movies)
        self.hidden_bias = torch.randn(1,self.number_of_hidden_nodes)
        self.visible_bias = torch.randn(1,self.number_of_movies)
        self.no_epochs = no_epochs
        
        if(debug):
            print("---------------Input Weights and bias begin----------------------\n")
            print(" Weights : {0} \n Hidden bias : {1} \n Visible bias : {2}".format(self.weights,self.hidden_bias,self.visible_bias))
            print("---------------Input Weights and bias ends----------------------\n")
            
    def sample_hidden_for_given_visible(self,input_visible):
        wx = torch.mm(input_visible,self.weights.t())
        activation = wx + self.hidden_bias.expand_as(wx)
        prob_hidden_for_given_visible = torch.sigmoid(activation)
        return prob_hidden_for_given_visible,torch.bernoulli(prob_hidden_for_given_visible)
    
    def sample_visible_for_given_hidden(self,input_hidden):
        wy = torch.mm(input_hidden,self.weights)
        activation = wy + self.visible_bias.expand_as(wy)
        prob_visible_for_given_hidden = torch.sigmoid(activation)
        return prob_visible_for_given_hidden,torch.bernoulli(prob_visible_for_given_hidden)
    
    def train(self,visible_initial,visible_sampled,hidden_initial,hidden_sampled):
        self.weights += torch.mm(hidden_initial.t(),visible_initial) - torch.mm(hidden_sampled.t(),visible_sampled)
        self.hidden_bias += torch.sum((hidden_initial-hidden_sampled),0)
        self.visible_bias += torch.sum((visible_initial-visible_sampled),0)
        
    def trainRBM(self):
        
        for epoch in range(1,self.no_epochs+1):
            train_loss = 0
            s = 0
            for id_users in range(0,self.number_of_users-self.batch_size,self.batch_size):
                visible_initial = self.training_set[id_users:id_users+self.batch_size]
                visible_sampled = self.training_set[id_users:id_users+self.batch_size]
                prob_hidden_initial,_ = self.sample_hidden_for_given_visible(visible_initial)
                for i in range(10):
                    _,hidden_sampled = self.sample_hidden_for_given_visible(visible_sampled)
                    _,visible_sampled = self.sample_visible_for_given_hidden(hidden_sampled)
                    visible_sampled[visible_initial < 0] = visible_initial[visible_initial < 0]
                prob_hidden_smpled,_ = self.sample_hidden_for_given_visible(visible_sampled)
                self.train(visible_initial,visible_sampled,prob_hidden_initial,prob_hidden_smpled)
                train_loss += torch.mean(torch.abs(visible_initial[visible_initial>=0] - visible_sampled[visible_initial>=0]))
                s+=1
            print('Epoch : {0} and loss : {1} \n '.format(str(epoch),str(train_loss/s)))
            
    def testRBM(self):
        test_loss = 0
        s = 0
        for id_user in range(self.number_of_users):
            visible_training = self.training_set[id_user:id_user+1]
            visible_testing = self.test_set[id_user:id_user+1]
            if len(visible_testing[visible_testing >= 0]) > 0:
                _,hidden_sampled = self.sample_hidden_for_given_visible(visible_testing)
                _,visible_sampled = self.sample_visible_for_given_hidden(hidden_sampled)
            test_loss += torch.mean((visible_training[visible_training >=0] - visible_testing[visible_training >=0]))
            s+=1
            print('Test loss : {0} \n '.format(str(test_loss/s)))

def main():
    global rbm
    rbm = RestrictedBoltzmannMachine('ml-100k/u1.base','ml-100k/u1.test','\t',200,100,10,True)
    rbm.trainRBM()
    rbm.testRBM()
    
if __name__ == '__main__':
    main()