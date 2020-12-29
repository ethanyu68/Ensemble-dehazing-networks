import argparse
import torch
from torch.autograd import Variable
import numpy as np
import time, math, glob
import scipy.io as sio
import cv2, os
from utils import *
#from patchify import patchify, unpatchify
#from tqdm import trange

parser = argparse.ArgumentParser(description="Pytorch DRRN Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
#parser.add_argument("--model", type=str, default="dense_encoderM_AT", help="model path")
#parser.add_argument("--ep", type=int, default=4, help="epoch path")
#parser.add_argument("--dataset", default="finetune/HZ", type=str, help="dataset path")
parser.add_argument("--dataset", default="./NH-HAZE_testHazy", type=str, help="dataset path")
parser.add_argument("--activation", default="no_relu", type=str, help='activation relu')


def get_image_for_save(img):
    img = img.data[0].numpy()

    img = img * 255.
    img[img < 0] = 0
    img[img > 255.] = 255.
    img = np.rollaxis(img, 0, 3)
    img = img.astype('uint8')
    return img

opt = parser.parse_args()
cuda = opt.cuda
save_path = os.path.join('EDN_results')
checkdirctexist(save_path)

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

model_path = os.path.join('EDN_3J_model.pth')
model = torch.load(model_path)["model"].module

image_list = glob.glob(os.path.join(opt.dataset,'*.png'))

avg_elapsed_time = 0.0
count = 0.0
for image_name in image_list:
    count += 1
    print("Processing ", image_name)
    img = cv2.imread(image_name)

    img = img.astype(np.float32)
    H,W,C = img.shape

    P = 512
    # patches = patchify(img, (P, P, 3), step=168)  # split image into 2*3 small 2*2 patches.
    print("\t\tBreak image into patches of {}x{}".format(P,P))
    # X,Y,_,_,_,_ = patches.shape
    # W = W_H[1]
    # H = W_H[0]
    Wk = W
    Hk = H
    if W % 32:
        Wk = W + (32 - W % 32)
    if H % 32:
        Hk = H + (32 - H % 32)


    # img_padded = np.zeros((Hk, Wk, 3))
    # img_padded[0:H, 0:W, : ]= img
    img = np.pad(img, ((0, Hk-H), (0,Wk-W), (0,0)), 'reflect')


    im_input = img/255.0
    im_input = np.expand_dims(np.rollaxis(im_input, 2),  axis=0)
    im_input_rollback = np.rollaxis(im_input[0], 0, 3)
    with torch.no_grad():
        # im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])
        im_input = Variable(torch.from_numpy(im_input).float())

        if cuda:
            model = model.cuda()
            #model.train(False)
            im_input = im_input.cuda()
        else:
            model = model.cpu()
        model.eval()
        start_time = time.time()
        im_output, _,_,_,_,_,_ = model(im_input, opt)
        elapsed_time = time.time() - start_time
        avg_elapsed_time += elapsed_time

    #im_output = im_output.cpu()
    im_output = im_output.cpu()
    im_output_forsave = get_image_for_save(im_output)
    #A_output = A.cpu()
    #A_output_forsave = get_image_for_save(A_output)
    #T_output = T.cpu()
    #T_output_forsave = get_image_for_save(T_output)
    #M_output = M.cpu()
    #M_output_forsave = get_image_for_save(M_output)
    # im_output_u = im_output_u.cpu()


    # im_output = im_output.data[0].numpy().astype(np.float32)
    # im_output = im_output.data[0].numpy()
    #
    # im_output = im_output*255.
    # im_output[im_output<0] = 0
    # im_output[im_output>255.] = 255.
    # im_output = np.rollaxis(im_output,0,3)

   
    # im_output_u_forsave = get_image_for_save(im_output_u)

    path, filename = os.path.split(image_name)

    im_output_forsave = im_output_forsave[0:H, 0:W, :]
    #A_output_forsave = A_output_forsave[0:H, 0:W, :]
    #T_output_forsave = T_output_forsave[0:H, 0:W, :]
    #M_output_forsave = M_output_forsave[0:H, 0:W, :]
    # im_output_forsave = unpatchify(patches=patches, imsize=(W,H,C))
    #cv2.imwrite(os.path.join(save_path, "resatt_{}".format(filename)), im_output_forsave)
    cv2.imwrite(os.path.join(save_path, "im_{}_{}".format(opt.ep, filename)), im_output_forsave)
    #cv2.imwrite(os.path.join(save_path, "A_{}_{}".format(opt.ep, filename)), A_output_forsave)
    #cv2.imwrite(os.path.join(save_path, "T_{}_{}".format(opt.ep, filename)), T_output_forsave)
    #cv2.imwrite(os.path.join(save_path, "M_{}_{}".format(opt.ep, filename)), M_output_forsave)
    # destRGB = cv2.cvtColor(im_output, cv2.COLOR_BGR2RGB)
    # cv2.imwrite(os.path.join(save_path, "u_{}_{}".format(opt.ep, filename)), im_output_u_forsave)

    # cv2.imwrite(os.path.join('val', 'INPUT_'+filename), im_input_rollback*255)
    # im_input_rollback = (im_input_rollback*255).astype('uint8')
    # destRGB_input = cv2.cvtColor(im_input_rollback, cv2.COLOR_BGR2RGB)
    # cv2.imwrite(os.path.join('val', 'INPUT_dest_' + filename), destRGB_input)



print("Dataset=", opt.dataset)
print("It takes average {}s for processing".format(avg_elapsed_time/count))

