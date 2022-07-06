
from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    # conv = import_ctx.import_from('basicblock', 'conv')
    # downsample_avgpool = import_ctx.import_from('basicblock','downsample_avgpool')
    # downsample_strideconv = import_ctx.import_from('basicblock','downsample_strideconv')
    # sequential = import_ctx.import_from('basicblock', 'sequential')
    # ResBlock = import_ctx.import_from('basicblock', 'ResBlock')
    # upsample_upconv = import_ctx.import_from('basicblock', 'upsample_upconv')
    # upsample_convtranspose = import_ctx.import_from('basicblock', 'upsample_convtranspose')
    single2tensor4 = import_ctx.import_from('basicblock', 'single2tensor4')
    uint2single = import_ctx.import_from('basicblock', 'uint2single')
    import bm3d
    from skimage.restoration import denoise_nl_means
    from functools import partial
    from abc import ABC, abstractmethod
    import torch
    import os
    import imageio
    import matplotlib.pyplot as plt
    import requests
    


class BaseDenoiser(ABC):
    def __init__(self) -> None:
        super().__init__()
        
    @abstractmethod
    def __call__(self, image, sigma):
        pass

class BaseDenoiser_drunet(ABC):
    def __init__(self) -> None:
        super().__init__()
        
    @abstractmethod
    def __call__(self, image, model, device):
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


#  : add denoiser optionnal parameters (eg dict input) ?

def load_denoiser(denoiser_name : str):
    denoiser_name_list = ['bm3d', 'nlm', 'drunet_gray']
    if denoiser_name == "bm3d":
        return BM3D_Denoiser()
    elif denoiser_name == "nlm":
        return NLM_Denoiser()
    elif denoiser_name == 'drunet_gray':
        download_pretrained_model()
        return model_drunet_class()
    else:
        raise ValueError('unknown denoiser name. Got {}, expected denoiser_name in {}'.format(denoiser_name, denoiser_name_list))






def download_pretrained_model(model_dir='model_zoo', model_name='drunet_gray.pth'):
    if os.path.exists(os.path.join(model_dir, model_name)):
        print(f'already exists, skip downloading [{model_name}]')
    else:
        os.makedirs(model_dir, exist_ok=True)
        if 'SwinIR' in model_name:
            url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{}'.format(model_name)
        else:
            url = 'https://github.com/cszn/KAIR/releases/download/v1.0/{}'.format(model_name)
        r = requests.get(url, allow_redirects=True)
        print(f'downloading [{model_dir}/{model_name}] ...')
        open(os.path.join(model_dir, model_name), 'wb').write(r.content)
        print('done!')

def model_drunet(img_L, model, device):
    print('Run model')
    img_L = uint2single(img_L) + np.random.normal(0, 15/255., img_L.shape)
    img_L = img_L.reshape((img_L.shape[0],img_L.shape[1],1))
    img_L = single2tensor4(img_L)
    img_L = torch.cat((img_L, torch.FloatTensor([15/255.]).repeat(1, 1, img_L.shape[2], img_L.shape[3])), dim=1)
    img_L = img_L.to(device)
    img_E = model(img_L)
    plt.figure()
    plt.subplot(211)
    plt.imshow(np.squeeze(img_L[0,0,:,:]))
    plt.subplot(212)
    plt.imshow(np.squeeze(img_E[0,:,:]))
    plt.show()
    
    return img_E

class model_drunet_class(BaseDenoiser_drunet):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, image, model, device):
        return model_drunet(image, model, device)
        





