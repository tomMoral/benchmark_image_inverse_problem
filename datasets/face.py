from benchopt import safe_import_context, BaseDataset


with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import misc
    import matplotlib.pyplot as plt
    make_blur = import_ctx.import_from('shared', 'make_blur')


class Dataset(BaseDataset):

    name = "Face"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # A * I + noise ~ N(mu, sigma)
    parameters = {
        'std_noise': [0.02],
        'size_blur': [27],
        'std_blur': [2.],
        'subsampling': [4],
        'type_A': ['deblurring'],
    }

    def __init__(self, std_noise=0.3,
                 size_blur=27, std_blur=8.,
                 subsampling=4,
                 random_state=27,
                 type_A='deblurring'):
        # Store the parameters of the dataset
        self.std_noise = std_noise
        self.size_blur = size_blur
        self.std_blur = std_blur
        self.subsampling = subsampling
        self.random_state = random_state
        self.type_A = type_A

    def set_A(self, img_shape):
        return make_blur(self.type_A, img_shape,
                         self.size_blur, self.std_blur)

    def get_data(self):
        rng = np.random.RandomState(self.random_state)
        img = misc.face(gray=True)[::self.subsampling, ::self.subsampling]
        img = img / 255.0
        A = self.set_A(img.shape)
        Y = (A @ img.flatten()).reshape(img.shape)
        Y += rng.normal(0, self.std_noise, size=img.shape)
        sigma_f = self.std_noise
        # plt.imshow(img)
        # plt.show()
        
        print(Y.shape)
        #Y = Y*255 + np.sqrt(2) * np.random.randn(img.shape[0], img.shape[1])
        data = dict(A=A, Y=Y, X_ref=img,sigma_f=sigma_f)
        return data
