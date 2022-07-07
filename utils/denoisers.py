
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import requests
    from pathlib import Path
    from abc import ABC, abstractmethod

    import bm3d
    import torch
    from skimage.restoration import denoise_nl_means

    single2tensor4 = import_ctx.import_from('basicblock', 'single2tensor4')
    uint2single = import_ctx.import_from('basicblock', 'uint2single')
    UNetRes = import_ctx.import_from('basicblock', 'UNetRes')


class BaseDenoiser(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, image, sigma):
        pass


class BM3DDenoiser(BaseDenoiser):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, image, sigma):
        return bm3d.bm3d(z=image, sigma_psd=sigma)


class NLMDenoiser(BaseDenoiser):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, image, sigma):
        return denoise_nl_means(image=image, sigma=sigma)


def download_pretrained_model(model_dir='model_zoo',
                              model_name='drunet_gray.pth'):

    model_dir = Path(model_dir)
    model_file = model_dir / model_name
    if not model_file.exists():
        model_dir.mkdir(exist_ok=True)
        if 'SwinIR' in model_name:
            url = (
                "https://github.com/JingyunLiang/SwinIR/releases/download/"
                f"v0.0/{model_name}"
            )
        else:
            url = (
                "https://github.com/cszn/KAIR/releases/download/"
                f"v1.0/{model_name}"
            )
        r = requests.get(url, allow_redirects=True)
        print(f'downloading [{model_dir}/{model_name}] ...')
        model_file.write_bytes(r.content)
        print('done!')

    return model_file


class DrunetDenoiser(BaseDenoiser):
    def __init__(self) -> None:
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        n_channels = 1

        model_path = download_pretrained_model()
        torch.cuda.empty_cache()
        model = UNetRes(
            in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512],
            nb=4, act_mode='R', downsample_mode="strideconv",
            upsample_mode="convtranspose"
        )
        model.load_state_dict(torch.load(model_path), strict=True)
        model = model.to(device=self.device)
        model.eval()

        self.model = model

    def __call__(self, image, sigma):
        # img_L = uint2single(image)
        # img_L = img_L.reshape((img_L.shape[0],img_L.shape[1],1))
        # img_L = single2tensor4(img_L)
        # img_L = img_L.to(device)
        with torch.no_grad():
            img_L = torch.cat([image, 15/255 * torch.ones_like(image)], dim=0).unsqueeze(0)
            return self.model(img_L).squeeze()


#  : add denoiser optionnal parameters (eg dict input) ?

def load_denoiser(denoiser_name: str):
    denoiser_name_list = ['bm3d', 'nlm', 'drunet_gray']
    if denoiser_name == "bm3d":
        return BM3DDenoiser()
    elif denoiser_name == "nlm":
        return NLMDenoiser()
    elif denoiser_name == 'drunet_gray':
        return DrunetDenoiser()
    else:
        raise ValueError(
            'unknown denoiser name. Got {}, expected denoiser_name in {}'
            .format(denoiser_name, denoiser_name_list)
        )
