from pathlib import Path
from joblib import load
from PIL.Image import open, ANTIALIAS
from PIL.ImageOps import fit
from numpy import array, ravel


class Bucket:
    def __init__(self, model_path, img_dim=None):
        self._model_path = model_path
        self._model = load(model_path)
        if img_dim is None:
            self._img_dim = (512, 512)
        else:
            self._img_dim = img_dim

    @property
    def model_path(self):
        return self._model_path

    @property
    def img_dim(self):
        return self._img_dim

    @property
    def model(self):
        return self._model

    def _read_image(self, image_path):
        return open(image_path)

    def _resize_image(self, img):
        return fit(img, self.img_dim, ANTIALIAS)

    def _engineer_features(self, img):
        return ravel(array(img)).reshape((1, -1))

    def _predict_bucket(self, x):
        return self.model.predict(x)[0]

    def classifyBucket(self, image_path):
        image_path = Path(image_path)
        assert image_path.exists()

        img = self._read_image(image_path)
        img = self._resize_image(img)
        x = self._engineer_features(img)

        return self._predict_bucket(x)
