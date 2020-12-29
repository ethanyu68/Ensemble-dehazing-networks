import numpy as np
import math, os


def checkdirctexist(dirct):
	if not os.path.exists(dirct):
		os.makedirs(dirct)


def PSNR_self(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = (pred -gt)
    rmse = math.sqrt(np.mean(imdff.cpu().data[0].numpy() ** 2))
    if rmse == 0:
        return 100
    return 20.0 * math.log10(1.0/rmse)

