from benchopt import BaseDataset, safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    make_blur = import_ctx.import_from('shared', 'make_blur')


class Dataset(BaseDataset):

    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # A * I + noise ~ N(mu, sigma)
    parameters = {
        'img_size': [64],
        'std_noise': [0.02],
        'size_blur': [27],
        'std_blur': [2.],
        'type_A': ['deblurring'],
    }

    def __init__(self, img_size=32, std_noise=0.3,
                 size_blur=27, std_blur=8.,
                 subsampling=4,
                 random_state=27,
                 type_A='deblurring'):
        # Store the parameters of the dataset
        self.img_size = img_size
        self.std_noise = std_noise
        self.size_blur = size_blur
        self.std_blur = std_blur
        self.subsampling = subsampling
        self.random_state = random_state
        self.type_A = type_A

    def get_data(self):
        rng = np.random.RandomState(self.random_state)
        img = rng.randn(self.img_size, self.img_size)
        img = img / 255.0
        filt, A = make_blur(
            self.type_A, img.shape, self.size_blur, self.std_blur
        )
        Y = (A @ img.flatten()).reshape(img.shape)
        Y += rng.normal(0, self.std_noise, size=img.shape)

        return dict(filt=filt, A=A, Y=Y, X_ref=img, sigma_f=self.std_noise)
