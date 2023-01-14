# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Adapted by Sofie DaniÃ«ls for the final project in the Course 'Deep Learning' (2022) at ETH Zurich

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional

import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy
import logging

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

def generate_images(
    network_pkl: str,
    outdir: str,
    projected_w: Optional[str],
):

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # Synthesize the result of a W projection.
    results = []
    # If we have a list of .npz file locations, continue
    if projected_w is not None:
        #logging.info('len projected w in generate ' + str(len(projected_w)))
        #logging.info('Generating images from projected W')
        # Loop through all locations and load latent vectors
        #logging.info(str(projected_w))
        for proj_w, filen in projected_w:
            #logging.info(str(proj_w))
            ws = np.load(proj_w)
            ws = ws['w']
            #logging.info('ws w shape ' + str(ws.shape))
            ws = torch.tensor(ws, device=device) # pylint: disable=not-callable
            #logging.info('ws shape[1:] ' + str(ws.shape[1:]))
            #logging.info('G num ws, w dim ' + str(G.num_ws) + " " + str(G.w_dim))
            assert ws.shape[1:] == (G.num_ws, G.w_dim)
            for idx, w in enumerate(ws):
                img = G.synthesis(w.unsqueeze(0), noise_mode='const')
                # From [-1,1] to [0, 255]
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                # Save image in outdir
                PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/proj_' + filen + '.png')
                # Add image to list
                results.append(img[0].cpu().numpy())

    return results

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
