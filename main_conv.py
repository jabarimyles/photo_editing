Main.py conv

from return_faces import face_detector
from calculate_blur import laplacian_variance
import numpy as np
from convolutions import filter
import matplotlib.pyplot as plt
from PIL import Image
import sys



def main(file):
    log_kernel = np.array([[-1,-1, -1], [-1,8,-1], [-1,-1,-1]])
    #log_conv = filter(log_kernel)

    detector = face_detector(file)
    faces    = detector.return_faces_as_images()
    log_conv = Image.open(file).convert('L')
    #log_conv.get_log(Image.open(file).convert('L'))
    #plt.imshow(log_conv.log_img, cmap='gray')
    plt.imshow(log_conv, cmap='gray')
    plt.show()
    #print(log_conv.variance)

    print(str(len(faces)) + ' faces detected...')
    
    for face in faces:
        log_conv.get_log(face)
        plt.imshow(log_conv.log_img , cmap='gray')
        plt.show()
        print(str(log_conv.variance))





# file = 'blur.jpg'
# main(file)
# file = 'clear.jpg'
# main(file)

# file = 'blur.jpg'
# main(file)
if __name__ == '__main__':
    img_fp = str(sys.argv[1])
    main(img_fp)
