import argparse, os, time
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dense_deep_3JW import Dense
from dataset import DatasetFromHdf5
from utils import *
#from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from Vgg16 import Vgg16
import cv2
import pytorch_msssim
import pdb
import glob
# Training settings
parser = argparse.ArgumentParser(description="Pytorch DRRN")
parser.add_argument("--batchSize", type=int, default=4, help="Training batch size")
parser.add_argument("--patchSize", type=int, default=256, help="Training patch size")
parser.add_argument("--traindata", default="./data2020/data_combine/256_total_combine_complete.h5", type=str, help="Training datapath")#SynthesizedfromN18_256s64
parser.add_argument("--valdataset", default="./NH-HAZE_testHazy", type=str, help="validation datapath")

parser.add_argument("--nEpochs", type=int, default=100, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate, Default=0.1")
parser.add_argument("--step", type=int, default=10, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default=5")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--aug", action="store_true", help="Use aug?")

parser.add_argument("--resume", default="model/dense_residual_deepModel_mask/model_epoch_710.pth", type=str, help="Path to checkpoint, Default=None")
parser.add_argument("--start-epoch", default=1, type = int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.01, help="Clipping Gradients, Default=0.01")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default=1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default=0.9")
parser.add_argument("--weight_decay", "--wd", default=1e-6, type=float, help="Weight decay, Default=1e-4")
parser.add_argument("--pretrained", default="", type=str, help='path to pretrained model, Default=None')
parser.add_argument("--activation", default="no_relu", type=str, help='activation relu')
parser.add_argument("--ID", default="deep_3JW", type=str, help='ID for training')

parser.add_argument("--model", default="dense", type=str, help="unet or drrn or runet")
parser.add_argument("--freeze", action="store_true", help="freeze parameter??")
parser.add_argument("--alter", action="store_true", help="alternation for training??")
parser.add_argument("--totalloss", action="store_true", help="True for total loss")
parser.add_argument("--coeff_totalloss", default=0.2, type = float,help="")
parser.add_argument("--coeff_J", default=1, type = float,help="")

def main():
    global opt, model
    opt = parser.parse_args()
    print(opt)

    save_path = os.path.join('.', "model", "{}_{}".format(opt.model, opt.ID))
    log_dir = './records/{}_{}/'.format(opt.model, opt.ID)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    cuda = opt.cuda
    if cuda  and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    # opt.seed = 4222
    coeff_mse = opt.coeff_totalloss
    coeff_J = opt.coeff_J
    print("Random Seed: ", opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    train_set = DatasetFromHdf5(opt.traindata, opt.patchSize, opt.aug)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

    print("===> Building model")
    if opt.model == 'dense':
        model = Dense()
    else:
        raise ValueError("no known model of {}".format(opt.model))
    criterion = nn.MSELoss()
    Absloss = nn.L1Loss()
    ssim_loss = pytorch_msssim.MSSSIM()

    #loss_var = torch.std()
    if opt.freeze:
        model.freeze_pretrained()


    print("===> Setting GPU")
    if cuda:
        model = torch.nn.DataParallel(model).cuda()
        criterion = criterion.cuda()
        Absloss = Absloss.cuda()
        ssim_loss = ssim_loss.cuda()
        #loss_var = loss_var.cuda()
        vgg = Vgg16(requires_grad=False).cuda()

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("===> loading checkpoint: {}".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("===> no checkpoint found at {}".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            pretrained_dict = torch.load(opt.pretrained)['model'].state_dict()
            print("===> load model {}".format(opt.pretrained))
            model_dict = model.state_dict()
            # filter out unnecessary keys
            pretrained_dict = {k: v for  k,v in pretrained_dict.items() if k in model_dict}
            print("\t...loaded parameters:")
            for k,v in pretrained_dict.items():
                print("\t\t+{}".format(k))
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            # weights = torch.load(opt.pretrained)
            # model.load_state_dict(weights['model'].state_dict())
        else:
            print("===> no model found at {}".format(opt.pretrained))

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)#weight_decay=opt.weight_decay


    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        
        # Evaluate validation dataset and save images
        if epoch%1==0:
            save_val_path = os.path.join('test',opt.model+'_'+opt.ID)
            checkdirctexist(save_val_path)
            image_list = glob.glob(os.path.join(opt.valdataset,'*.png'))
            for image_name in image_list:
                print("Processing ", image_name)
                img = cv2.imread(image_name)
                img = img.astype(np.float32)
                H,W,C = img.shape
                P = 512
                print("\t\tBreak image into patches of {}x{}".format(P,P))
    
                Wk = W
                Hk = H
                if W % 32:
                    Wk = W + (32 - W % 32)
                if H % 32:
                    Hk = H + (32 - H % 32)
                    img = np.pad(img, ((0, Hk-H), (0,Wk-W), (0,0)), 'reflect')
                    im_input = img/255.0
                    im_input = np.expand_dims(np.rollaxis(im_input, 2),  axis=0)
                    im_input_rollback = np.rollaxis(im_input[0], 0, 3)
                    with torch.no_grad():
                        im_input = Variable(torch.from_numpy(im_input).float())
                        im_input = im_input.cuda()
                        model.eval()
                        
                        J,J1,J2,J3,w1,w2,w3= model(im_input, opt)
                        im_output = J
                       
                    im_output = im_output.cpu()
                    im_output_forsave = get_image_for_save(im_output)
                    J1_output = J1.cpu()
                    J1_output_forsave = get_image_for_save(J1_output)
                    J2_output = J2.cpu()
                    J2_output_forsave = get_image_for_save(J2_output)
                    J3_output = J3.cpu()
                    J3_output_forsave = get_image_for_save(J3_output)
                    W1_output = w1.cpu()
                    W1_output_forsave = get_image_for_save(W1_output)
                    W2_output = w2.cpu()
                    W2_output_forsave = get_image_for_save(W2_output)
                    W3_output = w3.cpu()
                    W3_output_forsave = get_image_for_save(W3_output)
                    
                    path, filename = os.path.split(image_name)

                    im_output_forsave = im_output_forsave[0:H, 0:W, :]
                    J1_output_forsave = J1_output_forsave[0:H, 0:W, :]
                    J2_output_forsave = J2_output_forsave[0:H, 0:W, :]
                    J3_output_forsave = J3_output_forsave[0:H, 0:W, :]
                    W1_output_forsave = W1_output_forsave[0:H, 0:W, :]
                    W2_output_forsave = W2_output_forsave[0:H, 0:W, :]
                    W3_output_forsave = W3_output_forsave[0:H, 0:W, :]
                    
                    cv2.imwrite(os.path.join(save_val_path, "{}_IM_{}".format(epoch-1, filename)), im_output_forsave)
                    cv2.imwrite(os.path.join(save_val_path, "{}_J1_{}".format(epoch-1, filename)), J1_output_forsave)
                    cv2.imwrite(os.path.join(save_val_path, "{}_J2_{}".format(epoch-1, filename)), J2_output_forsave)
                    cv2.imwrite(os.path.join(save_val_path, "{}_J3_{}".format(epoch-1, filename)), J3_output_forsave)
                    cv2.imwrite(os.path.join(save_val_path, "{}_W1_{}".format(epoch-1, filename)), W1_output_forsave)
                    cv2.imwrite(os.path.join(save_val_path, "{}_W2_{}".format(epoch-1, filename)), W2_output_forsave)
                    cv2.imwrite(os.path.join(save_val_path, "{}_W3_{}".format(epoch-1, filename)), W3_output_forsave)
        train(training_data_loader, optimizer, model, criterion, Absloss, ssim_loss, epoch, vgg)
        save_checkpoint(model, epoch, save_path)
                   
        

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (0.4 ** (epoch  // opt.step))
    return lr

def train(training_data_loader, optimizer, model, criterion, Absloss, ssim_loss, epoch, vgg):

    # lr policy
    lr = adjust_learning_rate(optimizer, epoch-1)
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))

    model.train()
    total_iter = 0
    i = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1])
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()
        # outputs
        J,J1,J2,J3,w1,w2,w3 = model(input, opt)
        output = J
        
        features_target = vgg(target)
        features_output = vgg(output)
        #features_output2 = vgg(J1)
        #features_output3 = vgg(J2)
        #features_output = vgg(J3)
            
        #loss_vgg1 = (criterion(features_output1.relu2_2, features_target.relu2_2) + criterion(features_output1.relu1_2, features_target.relu1_2) + criterion(features_output1.relu3_3, features_target.relu3_3))/3
        #loss_vgg2 = (criterion(features_output2.relu2_2, features_target.relu2_2) + criterion(features_output2.relu1_2, features_target.relu1_2) + criterion(features_output2.relu3_3, features_target.relu3_3))/3
        #loss_vgg3 = (criterion(features_output3.relu2_2, features_target.relu2_2) + criterion(features_output3.relu1_2, features_target.relu1_2) + criterion(features_output3.relu3_3, features_target.relu3_3))/3
        loss_vgg = (criterion(features_output.relu2_2, features_target.relu2_2) + criterion(features_output.relu1_2, features_target.relu1_2) + criterion(features_output.relu3_3, features_target.relu3_3))/3
            
        target_gray = 0.299 * target[:,2:3,:,:] + 0.587 * target[:,1:2,:,:] + 0.114 * target[:,0:1,:,:]
        out_gray = 0.299 * J2[:,2:3,:,:] + 0.587 * J2[:,1:2,:,:] + 0.114 * J2[:,0:1,:,:]
            
        ###MSE L1 edge
        loss_mse1  = criterion(J1, target)
        loss_j1 = loss_mse1 #+ loss_l11 #+ 0.3*loss_vgg1
        
        loss_mse2  = criterion(J2, target)
        loss_ssim = 1.0 - ssim_loss(target_gray, out_gray + 1e-12)
        loss_j2 = 0.05*loss_ssim #+ 0.3*loss_vgg2
        
        loss_l1 = Absloss(J3, target)
        loss_mse3 = criterion(J3,target)
        loss_j3 = 0.05*loss_l1
        
        loss_mse = criterion(output, target)
        loss = loss_j1 + loss_j2 + loss_j3 + loss_mse + 0.4*loss_vgg #+ loss_vgg2 + loss_vgg3 + loss_vgg)
        ##set optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration%20 == 0:
            psnr_output = PSNR_self(output.clone(), target.clone())
            #psnr_out1 = PSNR_self(out1.clone(), target.clone())
            #psnr_out2 = PSNR_self(out2.clone(), target.clone())
            print("===> Epoch[{}]({}/{},lr:{:.8f}): loss_vgg:{:.6f},  Loss_mse:{:.6f},  Loss_j1:{:.6f},  loss_j2:{:.6f},  loss_j3:{:.6f}, psnr: {:.3f}" "".format(epoch, iteration, len(training_data_loader),lr,  loss_vgg,  loss_mse,loss_j1, loss_j2,loss_j3, psnr_output))
            
        total_iter += 1

def save_checkpoint(model, epoch, save_path):
    model_out_path = os.path.join(save_path, "model_epoch_{}.pth".format(epoch))
    state = {"epoch": epoch, "model": model}
    # check path status
    if not os.path.exists("model/"):
        os.makedirs("model/")
    # save model
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

def edge_loss(out,target):
    out = 1/3 * (out[:,0,:,:] + out[:,1,:,:] + out[:,2,:,:])
    
    out.unsqueeze_(1)
    target = 1/3 * (target[:,0,:,:] + target[:,1,:,:] + target[:,2,:,:])
    
    target.unsqueeze_(1)
    x_filter = np.reshape(np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),[1,1,3,3])
    weights_x = torch.from_numpy(x_filter).float().cuda()
    y_filter = np.reshape(np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),[1,1,3,3])
    weights_y = torch.from_numpy(y_filter).float().cuda()

    g1_x = torch.nn.functional.conv2d(out, weights_x, stride=1, padding=1, bias=None)
    g1_y = torch.nn.functional.conv2d(out, weights_y, stride=1, padding=1, bias=None)
    g2_x = torch.nn.functional.conv2d(target, weights_x, stride=1, padding=1, bias=None)
    g2_y = torch.nn.functional.conv2d(target, weights_y, stride=1, padding=1, bias=None)

    g_1 = torch.pow(g1_x, 2) + torch.pow(g1_y, 2)
    g_2 = torch.pow(g2_x, 2) + torch.pow(g2_y, 2)

    return torch.mean((g_1 - g_2).pow(2))

def get_image_for_save(img):
    img = img.data[0].numpy()

    img = img * 255.
    img[img < 0] = 0
    img[img > 255.] = 255.
    img = np.rollaxis(img, 0, 3)
    img = img.astype('uint8')
    return img

if __name__ == "__main__":
    main()
