# AUTOGENERATED! DO NOT EDIT! File to edit: 06_maxDist.ipynb (unless otherwise specified).

__all__ = ['wass']

# Cell

import numpy as np
import ot

def wass(a,b,n_bins):
    """This function computes an approximation of the 2-Wasserstein distance between two datasets.

    Parameters
    ----------
    a : array
        dataset
    b : array
        dataset
    n_bins : int
        controls number of bins used to create the histograms for the dataset.

    Returns
    -------
    float
        estimate of the 2-Wasserstein distance between the two data-sets `a` and `b`
    """
    # get the range of the data
    a_range = (a.min(), a.max())
    b_range = (b.min(), b.max())

    # get range for union of a and b
    x_min = min(a_range[0], b_range[0])
    x_max = max(a_range[1], b_range[1])

    # get histograms and bins
    a_h, bins_a = np.histogram(a, range=(x_min,x_max),bins=n_bins, density=True)
    b_h, bins_b = np.histogram(b, range=(x_min,x_max),bins=n_bins, density=True)

    # ensure bins_a and bins_b are equal
    assert (bins_a == bins_b).all()

    # get bin width
    width = bins_a[1] - bins_a[0]

    # normalise histograms so they sum to 1
    a_h *= width
    b_h *= width

    # get cost matrix
    n = len(bins_a) - 1
    M = ot.dist(bins_a[:-1].reshape((n,1)),bins_a[:-1].reshape((n,1)))

    # compute the 2-wasserstein distance using POT and return
    return np.sqrt(ot.emd2(a_h,b_h,M))