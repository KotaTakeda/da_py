import gc
import weakref

import numpy as np

from da.letkf import LETKF


def test_localization_does_not_keep_discarded_filter_alive():
    filt = LETKF(lambda x, dt: x, np.eye(2), np.eye(2))
    filt.initialize(np.ones((3, 2)))
    filt._locR(0)
    reference = weakref.ref(filt)
    del filt
    gc.collect()
    assert reference() is None
