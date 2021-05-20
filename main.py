
#---- Standard Packages
import os
import sys
import json
from pathlib import Path

#---- Non-Standard packages
from matplotlib import pyplot
from matplotlib.patches import Rectangle, Circle
from PIL import Image
import pandas as pd
import cv2

#---- Local Packages
from main.Face import Face
from main.Bucket import Bucket

class Workflow:
    def __init__(self):
        self.faceDetector = Face(min_pixels=100)
        self.bucketClassifier = Bucket(model_path=Path.cwd()/'models'/'random_forest.joblib',
                                       img_dim=(256, 256)
                                      )

    def tagImage(self, fp):
        print(os.listdir())
        print(os.path.dirname(fp))

        if not os.path.exists(fp):
            raise FileNotFoundError('Image file path does not exist: %s' %fp)
        img = self._readImage(fp)
        tags = {
                    'file_name'   : os.path.basename(fp),
                    'file_path'   : os.path.dirname(fp),
                    'bucket'      : None,
                    'num_people'  : None,
                    'bbox_faces'  : None,
                    'confidences' : None,
                    'blur_coefs'  : None,
                    'groom'       : None,
                    'bride'       : None
                }

        tags['bucket'] = self._classifyBucket(fp)

        face_tags = self._getFaces(img)
        tags['num_people'] = len(face_tags['box'])
        tags['bbox_faces'] = face_tags['box'].tolist()
        tags['confidences'] = face_tags['confidence'].tolist()
        tags['blur_coefs'] = face_tags['blurry_score'].tolist()
        print(tags)

        output_path = os.path.join(tags['file_path'], os.path.splitext(tags['file_name'])[0] + '.json')
        with open(output_path, 'w') as f:
            json.dump(tags, f)

    def _tagDir(self, dp):
        if not os.path.exists(dp):
            raise FileNotFoundError('Directory path does not exist: %s' %dp)
        jpg_files = [f for f in os.listdir(dp) if os.path.splitext(f)[-1].lower() == '.jpg']
        for image in jpg_files:
            fp = os.path.join(dp, image)
            self.tagImage(fp)

    def _readImage(self, fp):
        return pyplot.imread(fp)

    def _classifyBucket(self, fp):
        return self.bucketClassifier._classifyBucket(fp)

    def _getFaces(self, img):
        face_tags = self.faceDetector.detect(img)
        return face_tags

if __name__ == '__main__':
    path = sys.argv[1]
    workflow = Workflow()

    if not os.path.exists(path):
        raise FileNotFoundError('Image file path does not exist: %s' %path)
    elif os.path.isfile(path):
        workflow.tagImage(path)
    elif os.path.isdir(path):
        workflow._tagDir(path)
    else:
        print('File path not file or directory')


