
import cv2


import matplotlib.pyplot as plt

class laplacian_variance:
    
    def __init__(self, faces, cutoff=100):
        self.faces = faces
        self.idk_what_this_is = cv2.CV_64F
        self.cutoff = cutoff
        
        self.sort_faces_by_bluriness()
        
        
        
        
    def img_to_laplacian(self, face):
        lap_img = cv2.Laplacian(face, self.idk_what_this_is)
        return lap_img 
        
        
    def calculate_variance(self, lap_img):
        var = lap_img.var()
        return var
        
    def sort_faces_by_bluriness(self):
        sorted_images = {
                         'clear' : [] ,
                         'blurry': [] 
                         }
        
        
        for face in self.faces:
            lap = self.img_to_laplacian(face)
            
            variance = self.calculate_variance(lap)
            print(variance)
            
            if variance > cutoff:
                sorted_images['clear'].append(face)
            else:
                sorted_images['blurry'].append(face)
            
   

