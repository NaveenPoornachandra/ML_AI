import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class SOM(object):
    def __init__(self,h,w,dim_feat):
        
        self.shape = (h,w,dim_feat)
        self.som = np.zeros((h,w,dim_feat))
        self.L0 = 0.0
        self.lam = 0.0
        self.sigma0 = 0.0
        
    def train(self,data,L0,lam,sigma0,initializer=np.random.rand):
        
        self.L0 = L0
        self.lam = lam
        self.sigma0 = sigma0
        
        self.som = initializer(*self.shape)
        self.data = data
        
        for t in itertools.count():
        
            if self.sigma(t) < 1.0:
                break
        
            i_data = np.random.choice(range(len(data)))
            bmu = self.find_bmu(data[i_data])
            self.update_som(bmu,data[i_data],t)
            
    def update_som(self,bmu,input_vector,t):
        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                dist_to_bmu = np.linalg.norm((np.array(bmu) - np.array((y,x))))
                self.update_cell((y,x),dist_to_bmu,input_vector,t)
            
    def update_cell(self,cell,dist_to_bmu,input_vector,t):
        
        self.som[cell] += self.N(dist_to_bmu,t)*self.L(t)*(input_vector-self.som[cell])
        
    def N(self,dist_to_bmu,t):
        curr_sigma = self.sigma(t)
        return np.exp(-(dist_to_bmu**2)/(2*curr_sigma**2))
    
    def sigma(self,t):
        return self.sigma0*np.exp(-t/self.lam)
    
    
    def find_bmu(self,input_vec):
        list_bmu = []
        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                dist = np.linalg.norm((input_vec - self.som[y,x]))
                list_bmu.append(((y,x),dist))
        list_bmu.sort(key = lambda x: x[1])
        
        return list_bmu[0][0]
    
    def update_bmu(self,bmu,input_vector,t):
        self.som[bmu] += self.L(t)*(input_vector-self[bmu])
        
    def L(self,t):
        return self.L0*np.exp(-t/self.lam)
    
    def quant_err(self):
        """ 
            Computes the quantization error of the SOM.
            It uses the data fed at last training.
        """
        bmu_dists = []
        for input_vector in self.data:
            bmu = self.find_bmu(input_vector)
            bmu_feat = self.som[bmu]
            bmu_dists.append(np.linalg.norm(input_vector-bmu_feat))
        return np.array(bmu_dists).mean()
    
    def plot_data(self):
        for _ in range(3):
            i_data = np.random.choice(range(len(self.data)))
            plt.imshow(self.data[i_data], interpolation='nearest')
            plt.show()
        
    def plot_som(self):
        plt.imshow(self.som, interpolation='nearest')
        plt.show()

def main():
    """ square_data = np.random.rand(5000,2)
    som_square = SOM(20,20,2)
    frames_square = []
    som_square.train(square_data,L0=0.8,lam=1e2,sigma0=10)
    print("quantization error:", som_square.quant_err())"""
    
    img=mpimg.imread('NaveeCP.jpg')
    
    color_data = np.reshape(img,(26838,3))
    
    som_color = SOM(189, 142, 3)
    som_color.train(color_data,L0=0.8,lam=1e2,sigma0=20)
    print("quantization error:", som_color.quant_err())
    som_color.plot_data()
    som_color.plot_som()
    
    

if __name__ == '__main__':
    main()
    