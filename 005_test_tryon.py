from options.train_options import TrainOptions
from models.networks import ResUnetGenerator, load_checkpoint_parallel
import torch.nn as nn
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import cv2
from tqdm import tqdm


# from utils.fid_scores import fid_pytorch
from lpips import LPIPS
from datetime import datetime
import torch_fidelity
from torch.autograd import Variable
from datetime import datetime
from torch.nn import functional as F
from math import exp

opt = TrainOptions().parse()
os.makedirs('sample/test_tryon/'+opt.name, exist_ok=True)

def ssim_fn(img1, img2, window_size = 11, size_average = True):
    
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(window_size, channel):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    def _ssim(img1, img2, window, window_size, channel, size_average = True):
        mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
        mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    # assuming its a Dataset if not a Tensor
    if not isinstance(img1, torch.Tensor):
        img1 = torch.stack([s["image"]["I"] for s in iter(img1)], dim=0)
    if not isinstance(img2, torch.Tensor):
        img2 = torch.stack([s["image"]["I"] for s in iter(img2)], dim=0)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

# 先確保 img_tryon[iii] 和 real_image[iii] 轉為 0~1 之間的數值
def normalize_img(img):
    X_min = img.min()
    X_max = img.max()
    img = (img - X_min) / (X_max - X_min)  # 轉換到 0~1
    return img

def CreateDataset(opt):
    from data.aligned_dataset import AlignedDataset
    dataset = AlignedDataset()
    dataset.initialize(opt, mode='test')

    return dataset

torch.cuda.set_device(opt.local_rank)

device = torch.device(f'cuda:{opt.local_rank}')

opt.PBAFN_gen_checkpoint = 'C:\\Users\\User\\Desktop\\yein_VTON\\checkpoints\\flow\\PBAFN_tryon_gen_epoch_041.pth'
opt.warproot = 'C:\\Users\\User\\Desktop\\yein_VTON\\sample\\test_warping\\result\\test'
opt.segroot = 'c:\\Users\\User\\Desktop\\yein_VTON\\sample\\test_warping\\seg\\test'

test_data = CreateDataset(opt)
test_loader = DataLoader(test_data, batch_size=opt.batchSize, shuffle=False, num_workers=0, pin_memory=True)

gen_model = ResUnetGenerator(36, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d, opt=opt)

gen_model.train()
gen_model.cuda()
load_checkpoint_parallel(gen_model, opt.PBAFN_gen_checkpoint)

gen_model = gen_model.to(device)
gen_model.eval()

lpips_fn = LPIPS(net='vgg', verbose=False).to(device)
val_pred_y = []
lpips = []
ssim_scores = []
fake_images = []
real_images = []

with torch.no_grad():
    for ii, data in enumerate(tqdm(test_loader)):
        real_image = data['image'].cuda()
        clothes = data['cloth'].cuda()
        preserve_mask = data['preserve_mask3'].cuda()
        preserve_region = real_image * preserve_mask
        warped_cloth = data['warped_cloth'].cuda()
        warped_prod_edge = data['warped_edge'].cuda()
        arms_color = data['arms_color'].cuda()
        arms_neck_label= data['arms_neck_lable'].cuda()
        pose = data['pose'].cuda()
        real_image = data['image'].cuda()
        background_color = data['background_color'].cuda()

        gen_inputs = torch.cat([preserve_region, warped_cloth, warped_prod_edge, arms_neck_label, arms_color, pose], 1)

        gen_outputs = gen_model(gen_inputs)
        p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        warped_cloth_mask = (warped_cloth > 0.2).float()
        m_composite = m_composite * warped_cloth_mask
        preserve_rendered = p_rendered * (1 - m_composite)
        # 獲取背景顏色並填補空缺
        filled_background = background_color * m_composite
        preserve_rendered += filled_background 
        p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)
        k = p_tryon

        bz = pose.size(0)
        for bb in range(bz):
            cloth_id = data['cloth_id'][bb]
            person_id = data['person_id'][bb]
            combine = k[bb].squeeze()
        
            cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy()+1)/2
            rgb = (cv_img*255).astype(np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            save_path = 'C:\\Users\\User\\Desktop\\yein_VTON\\sample\\test_tryon\\result\\'+person_id+'___'+cloth_id[:-4]+'.png'
            path_fid = 'C:\\Users\\User\\Desktop\\yein_VTON\\sample\\test_tryon\\fake_images\\'+person_id
            os.makedirs('C:\\Users\\User\\Desktop\\yein_VTON\\sample\\test_tryon\\result\\', exist_ok = True)
            os.makedirs('C:\\Users\\User\\Desktop\\yein_VTON\\sample\\test_tryon\\fake_images\\', exist_ok = True)

            cv2.imwrite(save_path, bgr)
            cv2.imwrite(path_fid, bgr)

            # 評估
            # fake_img = normalize_img(combine[bb]).to(device)   # (C, H, W)
            # real_img = normalize_img(real_image[bb]).to(device)   # (C, H, W)

            fake_img = combine.to(device)
            real = real_image[bb].to(device)
            
            # img_tryon_out = cv2.cvtColor(img_tryon_np, cv2.COLOR_RGB2BGR)
            # cv2.imwrite("C:\\Users\\User\\Desktop\\yein_VTON\\sample\\test_tryon\\fake_images", img_tryon_out)
            
            fake_images.append(fake_img.cpu())
            real_images.append(real.cpu())

            val_pred_y.append(combine[bb].detach().cpu())
            lpips_score = lpips_fn(fake_img.to(device), real.to(device)).detach().cpu().item()
            lpips.append(lpips_score)

            # print(fake_img.shape)

            ssim_value = ssim_fn(fake_img.unsqueeze(0), real.unsqueeze(0), size_average=True)
            ssim_scores.append(ssim_value.item())

            torch.cuda.empty_cache()  # 釋放記憶體

    metrics = torch_fidelity.calculate_metrics(
        input1="C:\\Users\\User\\Desktop\\yein_VTON\\sample\\test_tryon\\fake_images", # fake images
        input2="D:\\VITON-HD\\VITON-HD1024_Origin\\test\\image", # real images
        fid=True,
    )
    fid_score = metrics['frechet_inception_distance']

        
    val_pred_y = torch.cat(val_pred_y, dim=0)
    lpips = torch.tensor(lpips)
    print(datetime.now(), "- LPIPS: %.3f (+/- %.3f)" % (lpips.mean(), lpips.std()))

    fid_score = metrics['frechet_inception_distance']
    print(datetime.now(), "- FID: %.3f" % fid_score)

    # 輸出 SSIM 結果
    ssim_scores = torch.tensor(ssim_scores)
    print(datetime.now(), "- SSIM: %.3f (+/- %.3f)" % (ssim_scores.mean(), ssim_scores.std()))


