"""Distopia utilities
=======================
"""

__all__ = ('compute_affine_transform', )

import numpy as np


def compute_affine_transform(fixed, moving):
    '''Compute the affine transform by point set registration.
    The affine transform is the composition of a translation and a linear map.
    The two ordered lists of points must be of the same length larger or equal to 3.
    The order of the points in the two list must match.

    The 2D affine transform :math:`\mathbf{A}` has 6 parameters (2 for the translation and 4 for the linear transform).
    The best estimate of :math:`\mathbf{A}` can be computed using at least 3 pairs of matching points. Adding more
    pair of points will improve the quality of the estimate. The matching pairs are usually obtained by selecting
    unique features in both images and measuring their coordinates.
    :param list fixed: a list of the reference points.
    :param list moving: a list of the moving points to register on the fixed point.
    :returns translation, linear_map: the computed translation and linear map affine transform.

    Based on ``pymicro.view.vol_utils.compute_affine_transform``
    '''
    assert len(fixed) == len(moving)
    assert len(fixed) >= 3

    fixed_centroid = np.average(fixed, 0)
    moving_centroid = np.average(moving, 0)

    # offset every point by the center of mass of all the points in the set
    fixed_from_centroid = fixed - fixed_centroid
    moving_from_centroid = moving - moving_centroid
    covariance = moving_from_centroid.T.dot(fixed_from_centroid)
    variance = moving_from_centroid.T.dot(moving_from_centroid)

    # compute the full affine transform: translation + linear map
    linear_map = np.linalg.inv(variance).dot(covariance).T
    translation = fixed_centroid - linear_map.dot(moving_centroid)

    invt = np.linalg.inv(linear_map)
    rotation_scale = np.zeros((3, 3))
    rotation_scale[0:2, 0:2] = invt
    rotation_scale[2, 2] = 1

    trans = np.eye(3)
    trans[:2, 2] = -np.dot(invt, translation)
    mat = np.dot(trans, rotation_scale)

    # A multiplies fixed to get moving
    return mat
