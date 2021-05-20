from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from matplotlib.patches import Rectangle, Circle
from PIL import Image
import pandas as pd
import os
import sys
import cv2

class Face:
    def __init__(self, min_pixels=100):
        self.detector = MTCNN(min_face_size=min_pixels)
        self.min_pixels = min_pixels

    def getFacialObjectsMtcnn(self, pixels):
        faces = self.detector.detect_faces(pixels)
        return faces

    def saveLabeledImage(self,pixels, faces, output_fn):
        pyplot.imshow(pixels)
        ax = pyplot.gca()
        # plot each box
        for face in faces:
            # get coordinates
            x, y, width, height = face['box']
            # create the shape
            rect = Rectangle((x, y), width, height, fill=False, color='red')
            # draw the box
            ax.add_patch(rect)
            # draw the dots
            items_to_plot = ['left_eye', 'right_eye']
            for face_obj in items_to_plot:
                # create and draw dot
                loc = face['keypoints'][face_obj]
                dot = Circle(loc, radius=2, color='red')
                ax.add_patch(dot)
        # show the plot
        pyplot.savefig(output_fn)
        pyplot.clf()

    def getLPVariance(self, pixels, faces):
        variances = []
        for face in faces:
            x, y, w, h = face['box']
            face_im = pixels[y:y+h, x:x+h]
            var = cv2.Laplacian(face_im, cv2.CV_64F).var()
            variances.append(var)
        return variances

    def detect(self, img):
        faces = self.getFacialObjectsMtcnn(img)
        faces_df = pd.DataFrame(faces)
        faces_df['blurry_score'] = self.getLPVariance(img, faces)
        keep_cols = ['box', 'confidence', 'blurry_score']
        return faces_df[keep_cols]

if __name__ == '__main__':
    dir_name = sys.argv[1]
    face = Face(dir_name)
    face.runOnDir()

