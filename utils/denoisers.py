
from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    conv = import_ctx.import_from('basicblock', 'conv')

    downsample_avgpool = import_ctx.import_from('basicblock','downsample_avgpool')
    downsample_strideconv = import_ctx.import_from('basicblock','downsample_strideconv')
    sequential = import_ctx.import_from('basicblock', 'sequential')
    ResBlock = import_ctx.import_from('basicblock', 'ResBlock')
    upsample_upconv = import_ctx.import_from('basicblock', 'upsample_upconv')
    upsample_convtranspose = import_ctx.import_from('basicblock', 'upsample_convtranspose')
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
        return model_drunet_class()
    else:
        raise ValueError('unknown denoiser name. Got {}, expected denoiser_name in {}'.format(denoiser_name, denoiser_name_list))



class UNetRes(torch.nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(UNetRes, self).__init__()
        
        self.m_head = conv(in_nc, nc[0], bias=False, mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = sequential(*[ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = sequential(*[ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = sequential(*[ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.m_body  = sequential(*[ResBlock(nc[3], nc[3], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'), *[ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up2 = sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'), *[ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up1 = sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'), *[ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        self.m_tail = conv(nc[0], out_nc, bias=False, mode='C')

    def forward(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1)

        return x




def download_pretrained_model(model_dir='model_zoo', model_name='drunet_gray.pth'):
    if os.path.exists(os.path.join(model_dir, model_name)):
        #print(f'already exists, skip downloading [{model_name}]')
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

def model_drunet(img_L):
    
    download_pretrained_model()
    n_channels = 1
    model_pool = 'model_zoo'
    model_name = 'drunet_gray' 
    model_path = os.path.join(model_pool, model_name+'.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    model = UNetRes(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for _, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    noise_level_model = 15
    img_L = uint2single(img_L) + np.random.normal(0, noise_level_model/255., img_L.shape)
    img_L = img_L.reshape((img_L.shape[0],img_L.shape[1],1))
    img_L = single2tensor4(img_L)
    img_L = torch.cat((img_L, torch.FloatTensor([noise_level_model/255.]).repeat(1, 1, img_L.shape[2], img_L.shape[3])), dim=1)
    img_L = img_L.to(device)
    img_E = model(img_L)
    return img_E.numpy()

class model_drunet_class(BaseDenoiser):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, image, sigma):
        return model_drunet(image)





