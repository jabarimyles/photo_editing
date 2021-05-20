import numpy as np
import matplotlib.pyplot as plt

class filter:
    def __init__(self, kernel):
        self.kernel = kernel
        self.img=None
        self.padded_img = None
        self.filtered_img = None
        self.variance = None
        self.log_img = None
        
    
    
    def get_log(self, img):
        self.img = img
        filtered_img = self.convolve()
        self.subtract_log()
        self.getVariance()
        
        
        
    
    def pad(self):
        self.padded_img = np.pad(self.img, pad_width=1, mode='constant', constant_values=0)
        
        
    def convolve(self):
        self.pad()
        filtered_img = []
        for row in range(self.padded_img.shape[0] - self.kernel.shape[0] + 1):
        
            filtered_img.append([])
            for col in range(self.padded_img.shape[1] - self.kernel.shape[1] + 1):
                
                conv = self.padded_img[row: row + self.kernel.shape[0], col: col + self.kernel.shape[1]]
                #filtered_conv = np.sum(conv * kernel)
                filtered_conv = np.amin(conv)
                filtered_img[row].append(filtered_conv)

        filtered_img = np.array(filtered_img)
        self.filtered_img = filtered_img
    
    def subtract_log(self):
        self.log_img =  self.img - self.filtered_img
        
    def getVariance(self):
        self.variance = np.var(self.log_img)
    
    
