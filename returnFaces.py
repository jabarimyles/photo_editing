import cv2
import dlib
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



class face_detector:
    def __init__(self, path):
        self.path = path
        self.image = self.read_img(self.path)
        self.face_locations = self.find_faces()
        return None


    def read_img(self, path):
        return cv2.imread(path, 0)
        #return io.imread(path)


    def find_faces(self):
        face_detector = dlib.get_frontal_face_detector()
        return face_detector(self.image, 1)
        #print(detected_faces)

    def return_faces_as_images(self):
        images = []
        photo = self.image.tolist()

        for face in self.face_locations:
            image = []
            for row in range(face.top(), face.bottom() + 1):
                rows = []
                for col in range(face.left(), face.right() + 1):
                    rows.append(photo[row][col])
                image.append(rows)
            images.append(np.array(image))

        return images
