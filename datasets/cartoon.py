from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from PIL import Image, ImageOps
    import download
    import os

    make_blur = import_ctx.import_from("shared", "make_blur")

URL = "https://archive.org/download/SitaStills"
FILES = [
    "01.RamShootsDemons.png",
    "01.RishisBIG.png",
    "01.SitaRamForest.png",
    "02.RavSitaChariotSunset.png",
    "03.HanuBurnsLankaBIG.png",
    "04.Battlefield.png",
    "04.RamaArmyBIG.png",
    "05.RamSitaGods.png",
    "05.SitaHanuBanana.png",
    "06.RamHanuSitaRainReflect.png",
    "07.RamExilesSitaNoir.png",
    "07.SitaLaxmanChariotSwamp.png",
    "08.BlueLandscape.png",
    "09.SitaCriesARiver.png",
    "10RunToEarth.png",
]


class Dataset(BaseDataset):

    name = "Cartoon"

    install_cmd = "conda"
    requirements = ["pip:download"]

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # A * I + noise ~ N(mu, sigma)
    parameters = {
        "index": [0, 1, 2],
        "std_noise": [0.02],
        "size_blur": [27],
        "std_blur": [2.0],
        "subsampling": [10],
        "type_A": ["deblurring"],  # "denoising"],
        "type_n": [
            "gaussian",
        ],  # "laplace"],
    }

    def __init__(
        self,
        index=0,
        std_noise=0.02,
        size_blur=27,
        std_blur=8.0,
        subsampling=4,
        random_state=27,
        type_A="denoising",
        type_n="gaussian",
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
        img = download.download(
            os.path.join(URL, FILES[self.index]),
            os.path.join("./cartoon", FILES[self.index]),
            replace=False,
        )
        img = Image.open(img)
        img = (
            np.array(ImageOps.grayscale(img))[
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
        filt, A = make_blur(
            self.type_A, img.shape, self.size_blur, self.std_blur
        )
        Y = (A @ img.flatten()).reshape(img.shape) + n
        data = dict(filt=filt, A=A, Y=Y, X_ref=img, sigma_f=self.std_noise)

        return data
