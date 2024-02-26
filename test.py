import os,argparse
import numpy as np
from PIL import Image
from models.DwtFormer import DwtFormer
import torch
import torch.nn as nn
import torchvision.transforms as tfs 
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

abs=os.getcwd()+'/'
print(abs)

def tensorShow(tensors,titles=['haze']):
        fig=plt.figure()
        for tensor,tit,i in zip(tensors,titles,range(len(tensors))):
            img = make_grid(tensor)
            npimg = img.numpy()
            ax = fig.add_subplot(221+i)
            ax.imshow(np.transpose(npimg, (1, 2, 0)))
            ax.set_title(tit)
        plt.show()

parser=argparse.ArgumentParser()
parser.add_argument('--task',type=str,default='nhhaze',help='nhhaze or densehaze or ihaze or ohaze or SOTS')
parser.add_argument('--test_imgs',type=str,default='./datasets/XXX',help='Test imgs folder')
opt=parser.parse_args()
dataset=opt.task 

img_dir=abs+opt.test_imgs+'/'
output_dir=abs+f'dwtformer_{dataset}/'
print("pred_dir:",output_dir)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
model_dir='./trained_models/XXX.pk'

device='cuda' if torch.cuda.is_available() else 'cpu'
ckp=torch.load(model_dir,map_location=device)
net=DwtFormer()
net=nn.DataParallel(net)
net.load_state_dict(ckp['model'])
net.eval()

for im in os.listdir(img_dir):
    print(f'\r {im}',end='',flush=True)
    haze = Image.open(img_dir+im)
    haze1= tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])
    ])(haze)[None,::]
    haze_no=tfs.ToTensor()(haze)[None,::]
    
    with torch.no_grad():
        pred = net(haze1)
    ts=torch.squeeze(pred.clamp(0,1).cpu())
    vutils.save_image(ts,output_dir+im.split('.')[0]+'.png')
   