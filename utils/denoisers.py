import bm3d
from skimage.restoration import denoise_nl_means
from functools import partial
from abc import ABC, abstractmethod

class BaseDenoiser(ABC):
    def __init__(self) -> None:
        super().__init__()
        
    @abstractmethod
    def __call__(self, image, sigma):
        pass

class BM3D_Denoiser(BaseDenoiser):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, image, sigma):
        return bm3d.bm3d(z=image, sigma_psd=sigma)


class NLM_Denoiser(BaseDenoiser):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, image, sigma):
        return denoise_nl_means(image=image, sigma=sigma)


# TODO : add denoiser optionnal parameters (eg dict input) ?
def load_denoiser(denoiser_name : str):
    denoiser_name_list = ['bm3d', 'nlm']
    if denoiser_name == "bm3d":
        return BM3D_Denoiser()
    elif denoiser_name == "nlm":
        return NLM_Denoiser()
    else:
        raise ValueError('unknown denoiser name. Got {}, expected denoiser_name in {}'.format(denoiser_name, denoiser_name_list))


