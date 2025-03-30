# def get_tuple_shapes(my_tuple):
#     return tuple(
#         element.shape if hasattr(element, 'shape') else f"<{type(element).__name__}>"
#         for element in my_tuple
#     )


import torch
import torchvision

# 清理未使用的 CUDA 記憶體
torch.cuda.empty_cache()

# 檢查 PyTorch 是否安裝成功
print("PyTorch version:", torch.__version__)

# 檢查 CUDA 是否可用
print("CUDA Available:", torch.cuda.is_available())

# 確認 CUDA 版本
print("CUDA Version:", torch.version.cuda)
print(torchvision.__version__)

# 檢查 GPU 資訊
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
import triton
print("triton.__version__: ", triton.__version__)

'''顯示圖片'''
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import numpy as np
# # img_path = 'C:\\Users\\User\\Desktop\\yein_VTON\\sample\\test_warping\\ptexttp\\test\\03921_00.jpg___08015_00.png'
# img_path = 'D:\\VITON-HD\\VITON-HD1024_Origin\\test\\image-densepose\\00009_00.png'
# img = mpimg.imread(img_path)


# plt.imshow(img)
# plt.axis('off')  # 隱藏坐標軸
# plt.show()

'''確認最大、最小數字'''
# import os
# from PIL import Image
# import numpy as np
# from tqdm import tqdm

# def get_files(path):
#     files = os.listdir(path)
#     return files

# files = get_files('D:\\VITON-HD\\VITON-HD1024_Parsing\\test\\parse-bytedance')

# max_value = float('-inf')
# min_value = float('inf')
# for path in tqdm(files, desc="Processing images"):
#     # print(path)
#     img = Image.open(os.path.join('D:\\VITON-HD\\VITON-HD1024_Parsing\\test\\parse-bytedance', path))
#     image_gray = img.convert('L')
#     image_gray = np.array(image_gray)
#     x, y = image_gray.shape[0], image_gray.shape[1]
#     for i in range(x):
#         for j in range(y):
#             if(image_gray[i][j] < min_value):
#                 min_value = image_gray[i][j]
#             if(image_gray[i][j] > max_value):
#                 max_value = image_gray[i][j]

#     # pixels = list(image_gray.getdata())
#     # max_temp = max(pixels)
#     # min_temp = min(pixels)


#     # if(max_temp > max_value):
#     #     max_value = max_temp
#     # if(min_temp < min_value):
#     #     min_value = min_temp

#     # print('max: ', max_value, 'min: ', min_value)
#     # plt.imshow(image_gray)
#     # plt.axis('off')  # 隱藏坐標軸
#     # plt.show()
#     # break
# print('max: ', max_value, 'min: ', min_value)

'''參數量'''
# from models import AFWM
# from options.train_options import TrainOptions

# opt = TrainOptions().parse()

# warp_model = AFWM.AFWM_Vitonhd_lrarms(opt, 51)

# # total = sum([param.nelement() for param in warp_model.parameters()])
# # print('Number of parameter: %.2fM' % (total/1e6))


'''計算量'''
# import torch
# from thop import profile
# from models import AFWM
# from options.train_options import TrainOptions
# from fvcore.nn import FlopCountAnalysis, parameter_count_table

# opt = TrainOptions().parse()
# warp_model = AFWM.AFWM_Vitonhd_lrarms(opt, 51)
# warp_model.train()
# warp_model.cuda()
# device = torch.device(f'cuda:{opt.local_rank}')
# model = warp_model.to(device)

# concat = torch.randn(4, 51, 512, 384)
# clothes = torch.randn(4, 3, 512, 384)
# pre_clothes_edge = torch.randn(4, 1, 512, 384)
# cloth_parse_for_d = torch.randn(4, 1, 512, 384)
# clothes_left = torch.randn(4, 3, 512, 384)
# clothes_torso = torch.randn(4, 3, 512, 384)
# clothes_right = torch.randn(4, 3, 512, 384)
# left_cloth_sleeve_mask = torch.randn(4, 1, 512, 384)
# cloth_torso_mask = torch.randn(4, 1, 512, 384)
# right_cloth_sleeve_mask = torch.randn(4, 1, 512, 384)
# preserve_mask3 = torch.randn(4, 1, 512, 384)

# input = (concat, clothes, pre_clothes_edge, cloth_parse_for_d, clothes_left, 
#           clothes_torso, clothes_right, left_cloth_sleeve_mask, cloth_torso_mask, 
#           right_cloth_sleeve_mask, preserve_mask3)
# input = [tensor.to(device) for tensor in input]  # 如果 input 是張量列表

# flops, params = profile(model, inputs=(input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7], input[8], input[9], input[10]))
# flops = FlopCountAnalysis(model, input).total()
# params = parameter_count_table(model)

# print('flops: ', flops/1e9, 'params: ', params/1e6)
# print('params: ', params)


# from models.networks import AttU_Net

# model = AttU_Net(38, 3)
# total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"AttU_Net 的可訓練參數量: {total_params}")


from models.LightMUNet import LightMUNet
from options.train_options import TrainOptions
from models.networks import ResUnetGenerator

def count_parameters(model):
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()  # numel() returns the total number of elements in the tensor
    return total_params

# 創建你的 LightMUNet 模型實例
model = LightMUNet(spatial_dims=2, init_filters=8, in_channels=1, out_channels=2)

# 計算總參數量
total_params = count_parameters(model)
print(f"Total number of parameters in LightMUNet: {total_params}")



opt = TrainOptions().parse()
opt.warproot = 'C:\\Users\\User\\Desktop\\yein_VTON\\sample\\test_warping\\result\\train'
opt.segroot = 'c:\\Users\\User\\Desktop\\yein_VTON\\sample\\test_warping\\seg\\train'
# 創建 ResUnetGenerator 模型實例
model2 = ResUnetGenerator(input_nc=3, output_nc=3, num_downs=8, opt=opt)

# 計算總參數量
total_param2 = count_parameters(model2)
print(f"Total number of parameters in ResUnetGenerator: {total_param2}")