import numpy as np

def get_TV_filters():
    filt = np.array([
        [
            [0., 1., 0.],
            [0., -1., 0.],
            [0., 0., 0.]
        ],
        [
            [0., 0., 0.],
            [0., -1., 1.],
            [0., 0., 0.]
        ],
    ])
    return filt