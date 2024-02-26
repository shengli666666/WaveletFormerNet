from models.lossnet import LossNetwork
from models.DwtFormer import DwtFormer 
from models.PerceptualLoss import LossNetwork as PerLoss
import time, math, os
import numpy as np
from torch.backends import cudnn
from torch import optim
from tqdm import tqdm
import torch, warnings
from metrics import psnr, ssim
from torch import nn
warnings.filterwarnings('ignore')  
from option import opt, model_name, log_dir
from data_utils import dataset_select
from torchvision.models import vgg16
from torchvision.transforms import Resize
from loss import MS_SSIM_L1_LOSS

print('log_dir :', log_dir) 
print('model_name:', model_name)

models_ = {
            'dwtformer': DwtFormer(),
          }

start_time = time.time()  
T = opt.epochs  


def lr_schedule_cosdecay(t, T, init_lr=opt.lr):
    lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
    return lr


def train(net, loader_train, loader_test, optim, criterion):
    losses = []
    start_epoch = 0
    max_ssim = 0
    max_psnr = 0
    ssims = []
    psnrs = []
    my_loss = MS_SSIM_L1_LOSS()
    if opt.resume and os.path.exists(opt.model_dir):  
        print(f'resume from {opt.model_dir}') 
        ckp = torch.load(opt.model_dir)  
        losses = ckp['losses']  
        net.load_state_dict(ckp['model'])
        start_epoch = ckp['epoch']
        max_ssim = ckp['max_ssim']
        max_psnr = ckp['max_psnr']
        psnrs = ckp['psnrs']
        ssims = ckp['ssims']
        print(f'start_epoch:{start_epoch} start training ---')
    elif opt.refine :
    
        print(f'refine from {opt.refine_model_dir}')
        ckp = torch.load(opt.refine_model_dir)
        net.load_state_dict(ckp['model'])
        with torch.no_grad():  
            ssim_eval, psnr_eval = test(net, loader_test, max_psnr, max_ssim)  
            print(f'\nssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}')
    else:
        print('train from scratch *** ')
    # ----------------------------------------------------
    # -----------------------------------------------------
    for epoch in range(start_epoch + 1, opt.epochs + 1):
        net.train()
        lr = opt.lr
        if not opt.no_lr_sche:
            lr = lr_schedule_cosdecay(epoch, T)
            for param_group in optim.param_groups:  
                param_group["lr"] = lr
        for batch, data in enumerate(tqdm(loader_train)):
            x = data['haze']
            y = data['clear']
         
            x = x.to(opt.device)
            y = y.to(opt.device)  
            out= net(x)  
            loss = my_loss(out, y)
            if opt.perloss:  
                loss2 = criterion[1](out, y)
                loss = loss + 0.04 * loss2
            
            loss.backward()  

            optim.step()  
            optim.zero_grad()  
            losses.append(loss.item())  
        print(
            f'\rtrain loss : {np.mean(losses):.5f}| step :{epoch}/{opt.epochs}|lr :{lr :.7f} |time_used :{(time.time() - start_time) / 60 :.1f}',
            end='', flush=True)
        # -----------------------------------------------------------------------

        
        with torch.no_grad():  
            ssim_eval, psnr_eval = test(net, loader_test, max_psnr, max_ssim)  

            print(f'\nepoch :{epoch} |ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}')

          
            ssims.append(ssim_eval)
            psnrs.append(psnr_eval)
            if ssim_eval > max_ssim and psnr_eval > max_psnr:
                max_ssim = max(max_ssim, ssim_eval)
                max_psnr = max(max_psnr, psnr_eval)
                torch.save({
                    'epoch': epoch,
                    'max_psnr': max_psnr,
                    'max_ssim': max_ssim,
                    'ssims': ssims,
                    'psnrs': psnrs,
                    'losses': losses,
                    'model': net.state_dict()
                }, opt.model_dir)  
                print(f'\n model saved at epoch :{epoch}| max_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}\n' + opt.model_dir)
   
    np.save(f'./numpy_files/{model_name}_{opt.epochs}_losses.npy', losses)
    np.save(f'./numpy_files/{model_name}_{opt.epochs}_ssims.npy', ssims)
    np.save(f'./numpy_files/{model_name}_{opt.epochs}_psnrs.npy', psnrs)


def test(net, loader_test, max_psnr, max_ssim):
    net.eval() 
    
    ssims = []
    psnrs = []
    
    for i, data in enumerate(tqdm(loader_test)):
        inputs= data['haze']
        targets= data['clear']
        inputs = inputs.to(opt.device)
        targets = targets.to(opt.device)
        pred = net(inputs)
        if (pred.size() != targets.size()) :
            B, C, H, W = targets.size()
            pytorch_resize = Resize([H, W])
            pred = pytorch_resize(pred)
       
        ssim1 = ssim(pred, targets).item()
        psnr1 = psnr(pred, targets)
        ssims.append(ssim1)
        psnrs.append(psnr1)
    
    return np.mean(ssims), np.mean(psnrs)

# --- Define the perceptual loss network --- #
vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.to(opt.device)
for param in vgg_model.parameters():
    param.requires_grad = False

loss_network = LossNetwork(vgg_model)
loss_network.eval()

if __name__ == "__main__":  
    
    loader_train = dataset_select(opt.trainset, True, opt)
    loader_test = dataset_select(opt.trainset, False, opt)
    net = models_[opt.net]
    net = net.to(opt.device)
    if opt.device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True


    criterion = []
    criterion.append(nn.L1Loss().to(opt.device))  
    if opt.perloss:
        vgg_model = vgg16(pretrained=True).features[:16]  
        vgg_model = vgg_model.to(opt.device)
        for param in vgg_model.parameters():
            param.requires_grad = False  
        criterion.append(PerLoss(vgg_model).to(opt.device))  
    optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()), lr=opt.lr, betas=(0.9, 0.999),
                           eps=1e-08)
    
    optimizer.zero_grad()
    train(net, loader_train, loader_test, optimizer, criterion)

