from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import os
    import numpy as np
    from PIL import Image
    import download

    make_blur = import_ctx.import_from("shared", "make_blur")

SET11_NAMES = [
    'Monarch.tif', 
    'Parrots.tif', 
    'barbara.tif', 
    'boats.tif', 
    'cameraman.tif', 
    'fingerprint.tif', 
    'flinstones.tif', 
    'foreman.tif', 
    'house.tif', 
    'lena256.tif', 
    'peppers256.tif'
    ]
    
SET11_DIR = "https://github.com/jianzhangcs/ISTA-Net-PyTorch/blob/master/data/Set11"
SET11_DOWNLOAD_PATH = [os.path.join(SET11_DIR, '{}?raw=true'.format(name)) for name in SET11_NAMES]

class Dataset(BaseDataset):
    # classical grayscale dataset
    name = "set11"

    install_cmd = "conda"
    requirements = ["pip:download"]

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # A * I + noise ~ N(mu, sigma)
    parameters = {
        "std_noise": [0.02],
        "size_blur": [27],
        "std_blur": [2.0],
        "subsampling": [1],
        "type_A": ["deblurring"],  # "denoising"],
        "type_n": [
            "gaussian",
        ],  # "laplace"],
        "index" : list(range(11))
    }

    def __init__(
        self,
        std_noise=0.02,
        size_blur=27,
        std_blur=8.0,
        subsampling=4,
        random_state=27,
        type_A="denoising",
        type_n="gaussian",
        index=[0]
    ):
        # Store the parameters of the dataset
        self.std_noise = std_noise
        self.size_blur = size_blur
        self.std_blur = std_blur
        self.subsampling = subsampling
        self.random_state = random_state
        self.type_A, self.type_n = type_A, type_n

    def set_A(self, height):
        return make_blur(self.type_A, height, self.size_blur, self.std_blur)

    def get_data(self):
        rng = np.random.RandomState(self.random_state)
        img_remote_path = SET11_DOWNLOAD_PATH[self.index]
        img_name = SET11_NAMES[self.index]
        os.makedirs('data', exist_ok=True)
        image_local_path = 'data/set11/{}'.format(img_name)
        img = download.download(
            img_remote_path,
            image_local_path,
            replace=False,
        )
        img = Image.open(image_local_path, mode='r')
        #img = io.imread(image_local_path)
        img = (
            np.array(img)[
                :: self.subsampling, :: self.subsampling
            ]
        ) / 255.0
        height, width = img.shape
        if self.type_n == "gaussian":
            # noise ~ N(loc, scale)
            n = rng.normal(0, self.std_noise, size=(height, width))
        elif self.type_n == "laplace":
            # noise ~ L(loc, scale)
            n = rng.laplace(0, self.std_noise, size=(height, width))
        A = make_blur(
            self.type_A, img.shape, self.size_blur, self.std_blur
        )
        Y = (A @ img.flatten()).reshape(img.shape) + n
        data = dict(A=A, Y=Y, X_ref=img, sigma_f=self.std_noise)

        return data
