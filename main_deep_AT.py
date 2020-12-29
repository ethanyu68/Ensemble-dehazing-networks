import argparse, os, time
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dense_deep_residual_AT_adaptive import Dense
from dataset import DatasetFromHdf5
from utils import *
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from Vgg16 import Vgg16
import cv2
import glob
import pytorch_msssim
import pdb
# Training settings
parser = argparse.ArgumentParser(description="")

parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--aug", action="store_true", help="Use aug?")
parser.add_argument("--start-epoch", default=1, type = int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.01, help="Clipping Gradients, Default=0.01")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default=1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default=0.9")
parser.add_argument("--weight_decay", "--wd", default=1e-6, type=float, help="Weight decay, Default=1e-4")
parser.add_argument("--pretrained", default="", type=str, help='path to pretrained model, Default=None')
parser.add_argument("--activation", default="no_relu", type=str, help='activation relu')

parser.add_argument("--nEpochs", type=int, default=100, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate, Default=0.1")
parser.add_argument("--step", type=int, default=10, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default=5")
parser.add_argument("--valdataset", default="./valset2019", type=str, help="Training datapath")
parser.add_argument("--batchSize", type=int, default=4, help="Training batch size")
parser.add_argument("--patchSize", type=int, default=512, help="Training patch size")
parser.add_argument("--traindata", default=".h5", type=str, help="Training datapath")
parser.add_argument("--activation", default="no_relu", type=str, help='activation relu')
parser.add_argument("--ID", default="residual_deepModel_AT_actual_adaptive", type=str, help='ID of the saved model')
parser.add_argument("--model", default="dense", type=str, help="what model to use? default dense")

def main():
    global opt, model
    opt = parser.parse_args()
    print(opt)
    # check if folder for saving models exists or not. If not, create one.
    save_path = os.path.join('.', "model", "{}_{}".format(opt.model, opt.ID))
    log_dir = './records/{}_{}/'.format(opt.model, opt.ID)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # check if cuda is available
    cuda = opt.cuda
    if cuda  and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    # opt.seed = 4222
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
        if epoch%1==0:
            save_val_path = os.path.join('val',opt.model+'_'+opt.ID)
            checkdirctexist(save_val_path)
            image_list = glob.glob(os.path.join(opt.valdataset,'*.png'))
            for image_name in image_list:
                print("Processing ", image_name)
                
                img = cv2.imread(image_name)
             
                img = img.astype(np.float32)
                H,ww,C = img.shape
                
                P = 256
                print("\t\tBreak image into patches of {}x{}".format(P,P))
    
                Wk = ww
                Hk = H
                if ww % 32:
                    Wk = ww + (32 - ww % 32)
                if H % 32:
                    Hk = H + (32 - H % 32)
                    img = np.pad(img, ((0, Hk-H), (0,Wk-ww), (0,0)), 'reflect')
                    im_input = img/255.0
                    im_input = np.expand_dims(np.rollaxis(im_input, 2),  axis=0)
                    im_input_rollback = np.rollaxis(im_input[0], 0, 3)
                    with torch.no_grad():
                        im_input = Variable(torch.from_numpy(im_input).float())
                        im_input = im_input.cuda()
                        model = model.cuda()
                        model.eval()
                        
                        im_output, im_output1,im_output2, A1,A2,T,W,M= model(im_input, opt)
                        
                        
                    im_output = im_output.cpu()
                    im_output_forsave = get_image_for_save(im_output)
                    #im_output1_output = im_output1.cpu()
                    #im_output1_output_forsave = get_image_for_save(im_output1_output)
                    #im_output2_output = im_output2.cpu()
                    #im_output2_output_forsave = get_image_for_save(im_output2_output)
                    #W_output = W.cpu()
                    #W_output_forsave = get_image_for_save(W_output)
                    #M_output = M.cpu()
                    #M_output_forsave = get_image_for_save(M_output)
                    #EDGE_output = EDGE.cpu()
                    #EDGE_output_forsave = get_image_for_save(EDGE_output)

                    path, filename = os.path.split(image_name)

                    im_output_forsave = im_output_forsave[0:H, 0:ww, :]
                    #im_output1_output_forsave = im_output1_output_forsave[0:H, 0:ww, :]
                    #im_output2_output_forsave = im_output2_output_forsave[0:H, 0:ww, :]
                    #W_output_forsave = W_output_forsave[0:H, 0:ww, :]
                    #M_output_forsave = M_output_forsave[0:H, 0:ww, :]
		  
                    cv2.imwrite(os.path.join(save_val_path, "{}_IM_{}".format(epoch-1, filename)), im_output_forsave)
                    #cv2.imwrite(os.path.join(save_val_path, "{}_IM1_{}".format(epoch-1, filename)), im_output1_output_forsave)
                    #cv2.imwrite(os.path.join(save_val_path, "{}_IM2_{}".format(epoch-1, filename)), im_output2_output_forsave)
                    #cv2.imwrite(os.path.join(save_val_path, "{}_W_{}".format(epoch-1, filename)), W_output_forsave)
                    #cv2.imwrite(os.path.join(save_val_path, "{}_M_{}".format(epoch-1, filename)), M_output_forsave)
        train(training_data_loader, optimizer, model, criterion, Absloss, ssim_loss, epoch, vgg)
        save_checkpoint(model, epoch, save_path)
        # os.system("python eval.py --cuda --model=model/model_epoch_{}.pth".format(epoch))



def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (0.5 ** (epoch  // opt.step))
    return lr


def get_image_for_save(img):
    img = img.data[0].numpy()

    img = img * 255.
    img[img < 0] = 0
    img[img > 255.] = 255.
    img = np.rollaxis(img, 0, 3)
    img = img.astype('uint8')
    return img



def train(training_data_loader, optimizer, model, criterion, Absloss, ssim_loss, epoch, vgg):

    writer = SummaryWriter(log_dir='./records/{}_{}/'.format(opt.model, opt.ID))

    # lr policy
    lr = adjust_learning_rate(optimizer, epoch-1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))

    model.train()
    total_iter = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1])
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()

        # outputs
        output, output1,output2, A1,A2,T,W,M = model(input, opt)

        if opt.model == 'rcunet':
            # percetional features
            features_target = vgg(target)
            features_output_eh = vgg(output_eh)

            # variance
            loss_var_r = criterion(output_eh[:, 0:1, :, :].view(opt.batchSize,-1).std(dim=1), target[:, 0:1, :, :].view(opt.batchSize,-1).std(dim=1))
            loss_var_g = criterion(output_eh[:, 1:2, :, :].view(opt.batchSize,-1).std(dim=1), target[:, 1:2, :, :].view(opt.batchSize,-1).std(dim=1))
            loss_var_b = criterion(output_eh[:, 2:3, :, :].view(opt.batchSize,-1).std(dim=1), target[:, 2:3, :, :].view(opt.batchSize,-1).std(dim=1))

            # color
            loss_color_r = criterion(output_eh[:, 0:1, :, :], target[:, 0:1, :, :])
            loss_color_g = criterion(output_eh[:, 1:2, :, :], target[:, 1:2, :, :])
            loss_color_b = criterion(output_eh[:, 2:3, :, :], target[:, 2:3, :, :])

            # loss
            loss_mse = criterion(output_eh, target)
            loss_l1 = Absloss(output_eh, target)
            # loss_mse_eh = criterion(output_eh, target)
            loss_vgg = criterion(features_output_eh.relu2_2, features_target.relu2_2) + \
                         criterion(features_output_eh.relu1_2, features_target.relu1_2)
            # loss_vgg_eh = criterion(features_output_eh.relu2_2, features_target.relu2_2) + \
            #              criterion(features_output_eh.relu1_2, features_target.relu1_2)
            loss = loss_mse + loss_vgg + loss_l1

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            (loss_color_r + loss_var_r).backward(retain_graph=True)
            (loss_color_g + loss_var_g).backward(retain_graph=True)
            (loss_color_b + loss_var_b).backward()
            # Gradient Clipping
            # clip = opt.clip / lr
            # nn.utils.clip_grad_norm(model.parameters(), clip)
            optimizer.step()

        elif opt.model == 'rccunet' or opt.model == 'dense' or opt.model=='unet':
            features_target = vgg(target)
            features_output_eh = vgg(output)
            #haze = target*output2 + (1.0 - output2)*A
            loss_mse = criterion(output, target)
            loss_mse1 = criterion(output1, target)
            loss_mse2 = criterion(output2, target)
            loss_vgg = criterion(features_output_eh.relu2_2, features_target.relu2_2) + criterion(features_output_eh.relu1_2, features_target.relu1_2) + criterion(features_output_eh.relu3_3, features_target.relu3_3)
            loss_vgg = loss_vgg/3.0
            
            loss = loss_mse + 0.5*loss_mse1 + 0.5*loss_mse2 + 0.5*loss_vgg #+ .01*var_loss + 0.1*s_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        writer.add_scalar('data/scalar1', loss.item(), total_iter)

        if iteration%50 == 0:
            psnr_run = PSNR_self(output.clone(), target.clone())

            if not opt.model == 'rccunet':
                psnr_run_eh = PSNR_self(output.clone(), target.clone())

            input_display = vutils.make_grid(input, normalize=False, scale_each=True)
            writer.add_image('Image/train_input', input_display, total_iter)

            output_display = vutils.make_grid(output, normalize=False, scale_each=True)
            writer.add_image('Image/train_output', output_display, total_iter)

            if not opt.model == 'rccunet':
                output_display = vutils.make_grid(output, normalize=False, scale_each=True)
                writer.add_image('Image/train_output_eh', output_display, total_iter)

            gt_display = vutils.make_grid(target, normalize=False, scale_each=True)
            writer.add_image('Image/train_target', gt_display, total_iter)

            # psnr_run = 100
            if not opt.model == 'rccunet':
                print("===> Epoch[{}]({}/{}): Loss_mse: {:.10f} , Loss_vgg: {:.10f} , Loss_recon: {:.10f} ,ssim: {:.10f} ,psnr: {:.3f} | eh{:.3f}"
                      "".format(epoch, iteration, len(training_data_loader), loss_mse.data.item(), loss_vgg, loss_mse1, loss_mse2, psnr_run, psnr_run_eh))
            elif opt.model == 'rccunet':
                print("===> Epoch[{}]({}/{}): Loss_mse: {:.10f} , Loss_vgg: {:.10f} , psnr: {:.3f}"
                      "".format(epoch, iteration, len(training_data_loader), loss_mse.data[0], loss_vgg, psnr_run))

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

if __name__ == "__main__":
    main()
