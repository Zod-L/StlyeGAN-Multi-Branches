# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Frechet Inception Distance (FID) from the paper
"GANs trained by a two time-scale update rule converge to a local Nash
equilibrium". Matches the original implementation by Heusel et al. at
https://github.com/bioinf-jku/TTUR/blob/master/fid.py"""

import numpy as np
import scipy.linalg
from . import metric_utils

#----------------------------------------------------------------------------

def compute_fid(opts, max_real, num_gen):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.
    opts.cache = False


    glo, em, en = metric_utils.compute_feature_stats_for_dataset(
    opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
    rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_real)
    
    glo_mu_real, glo_sigma_real = glo.get_mean_cov()
    em_mu_real, em_sigma_real = em.get_mean_cov()
    en_mu_real, en_sigma_real = en.get_mean_cov()


    glo, em, en = metric_utils.compute_feature_stats_for_generator(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=1, capture_mean_cov=True, max_items=num_gen)



    glo_mu_gen, glo_sigma_gen = glo.get_mean_cov()
    em_mu_gen, em_sigma_gen = em.get_mean_cov()
    en_mu_gen, en_sigma_gen = en.get_mean_cov()



    if opts.rank != 0:
        return {"global":float('nan'), "eye_mouth":float('nan'), "ear_nose":float('nan')} 

    m_em = np.square(em_mu_gen - em_mu_real).sum()
    s_em, _ = scipy.linalg.sqrtm(np.dot(em_sigma_gen, em_sigma_real), disp=False) # pylint: disable=no-member
    fid_em = np.real(m_em + np.trace(em_sigma_gen + em_sigma_real - s_em * 2))

    m_en = np.square(en_mu_gen - en_mu_real).sum()
    s_en, _ = scipy.linalg.sqrtm(np.dot(en_sigma_gen, en_sigma_real), disp=False) # pylint: disable=no-member
    fid_en = np.real(m_em + np.trace(en_sigma_gen + en_sigma_real - s_en * 2))


    m_glo = np.square(glo_mu_gen - glo_mu_real).sum()
    s_glo, _ = scipy.linalg.sqrtm(np.dot(glo_sigma_gen, glo_sigma_real), disp=False) # pylint: disable=no-member
    fid_glo = np.real(m_glo + np.trace(glo_sigma_gen + glo_sigma_real - s_glo * 2))

    return {"global":float(fid_glo), "eye_mouth":float(fid_em), "ear_nose":float(fid_en)} 

#----------------------------------------------------------------------------
