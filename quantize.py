import os
import pdb
import glob
from pathlib import Path
import cv2
import copy
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, quantize_static, QuantType, CalibrationMethod
from lib.models import get_net
from lib.config import cfg
from lib.models.YOLOP import YOLOP
from lib.models.common2 import Conv, SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect, SharpenConv
from torch.nn import Upsample

device = 'cuda:0'
img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']
# 模型路径
model_fp32_pth = 'weights/End-to-end.pth'
model_fp32_onnx = 'weights/yolop-640-640.onnx'
model_quant_static = 'weights/yolop-640-640-quant.onnx'
images_folder_bdd = '/media/zhutianyi/KESU/datasets/bdd100k/images/100k/val'
images_folder_kitti = '/media/zhutianyi/KESU/datasets/kitti/testing/image_2'

def evaluate(model, val_data_dir_bdd, val_data_dir_kitti):
    p = str(Path(val_data_dir_bdd)) 
    p = os.path.abspath(p)
    image_names = os.listdir(p)
    image_list1 = [os.path.join(val_data_dir_bdd, image_name) for image_name in image_names if os.path.exists(os.path.join(val_data_dir_bdd, image_name))][:100]

    p = str(Path(val_data_dir_kitti)) 
    p = os.path.abspath(p)
    image_names = os.listdir(p)
    image_list2 = [os.path.join(val_data_dir_kitti, image_name) for image_name in image_names if os.path.exists(os.path.join(val_data_dir_kitti, image_name))][:100]
    image_list = image_list1 + image_list2
    
    for img_path in image_list:
        # img_path = '/media/zhutianyi/KESU/project/YOLOP/inference/images/000000.png'
        img_bgr = cv2.imread(img_path)
        # convert to RGB
        img_rgb = img_bgr[:, :, ::-1].copy()
        canvas, r, dw, dh, new_unpad_w, new_unpad_h = resize_unscale(img_rgb, (640, 640))
        img = canvas.copy().astype(np.float32)  # (3,640,640) RGB
        img /= 255.0
        img[:, :, 0] -= 0.485
        img[:, :, 1] -= 0.456
        img[:, :, 2] -= 0.406
        img[:, :, 0] /= 0.229
        img[:, :, 1] /= 0.224
        img[:, :, 2] /= 0.225
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)  # (1, 3,640,640)
        input = torch.from_numpy(img).float()
        
        a , b, c = model(input)
        # _, da_seg_mask = torch.max(b, 1)
        # import pdb;pdb.set_trace()
        # pass

def resize_unscale(img, new_shape=(640, 640), color=114):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    canvas = np.zeros((new_shape[0], new_shape[1], 3))
    canvas.fill(color)
    # Scale ratio (new / old) new_shape(h,w)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # w,h
    new_unpad_w = new_unpad[0]
    new_unpad_h = new_unpad[1]
    pad_w, pad_h = new_shape[1] - new_unpad_w, new_shape[0] - new_unpad_h  # wh padding

    dw = pad_w // 2  # divide padding into 2 sides
    dh = pad_h // 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)

    canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :] = img
    return canvas, r, dw, dh, new_unpad_w, new_unpad_h  # (dw,dh)

model = get_net(cfg)
checkpoint = torch.load("./model_float32.pth", map_location= device)
model.load_state_dict(checkpoint)
model.eval()

# ###
img_path = '/media/zhutianyi/KESU/project/YOLOP/inference/images/000000.png'
img_bgr = cv2.imread(img_path)
# convert to RGB
img_rgb = img_bgr[:, :, ::-1].copy()
canvas, r, dw, dh, new_unpad_w, new_unpad_h = resize_unscale(img_rgb, (640, 640))
img = canvas.copy().astype(np.float32)  # (3,640,640) RGB
#normalize
img /= 255.0
img[:, :, 0] -= 0.485
img[:, :, 1] -= 0.456
img[:, :, 2] -= 0.406
img[:, :, 0] /= 0.229
img[:, :, 1] /= 0.224
img[:, :, 2] /= 0.225
img = img.transpose(2, 0, 1)
img = np.expand_dims(img, 0)  # (1, 3,640,640)
input = torch.from_numpy(img).float()
# _,b,_ = model(input)
# _,da_seg_mask = torch.max(b, 1)
# import pdb;pdb.set_trace()
# ###


##fuse
fused_model  = copy.deepcopy(model)
# for module_name, module in fused_model.named_children():
#     for layer_index,module in module.named_children():
#         if YOLOP[int(layer_index)+1][1] is Conv:
#             torch.quantization.fuse_modules(module,[["conv", "bn", "act"]],inplace=True)
#         elif YOLOP[int(layer_index)+1][1] is Focus:
#             for module_name,module in module.named_children():
#                 if module_name=='conv':
#                     torch.quantization.fuse_modules(module,[["conv", "bn", "act"]],inplace=True)
#         elif YOLOP[int(layer_index)+1][1] is SPP:
#             for module_name,module in module.named_children():
#                 if module_name=='cv1' or module_name=='cv2':
#                     torch.quantization.fuse_modules(module,[["conv", "bn", "act"]],inplace=True)
#         elif YOLOP[int(layer_index)+1][1] is BottleneckCSP:
#             for module_name,module in module.named_children():
#                 if module_name=='cv1' or module_name=='cv4':
#                     torch.quantization.fuse_modules(module,[["conv", "bn", "act"]],inplace=True)
#                 elif module_name=='m':
#                     for module_name,module in module.named_children():
#                         for module_name,module in module.named_children():
#                             if module_name=='cv1' or module_name=='cv2':
#                                 torch.quantization.fuse_modules(module,[["conv", "bn", "act"]],inplace=True)
###


model_qconfig = torch.quantization.get_default_qconfig('fbgemm')
fused_model.qconfig = model_qconfig

model_fp32_prepared = torch.quantization.prepare(fused_model)

evaluate(model_fp32_prepared,images_folder_bdd,images_folder_kitti)
model_fp32_prepared.quant.activation_post_process.max_val = torch.tensor(2.64)
# model_fp32_prepared.quant.activation_post_process.max_val = torch.tensor(-2.1179)
model_int8 = torch.quantization.convert(model_fp32_prepared)
model_int8.eval()

state_dict = model_int8.state_dict()
keys_to_delete = []
# 遍历状态字典
for param_name, param_value in state_dict.items():
    if param_value is None:
        keys_to_delete.append(param_name)

for param_name in keys_to_delete:
    name_weight = '.'.join(param_name.split('.')[:-1]+['weight'])
    n = state_dict[name_weight].shape[0]
    state_dict[param_name] = torch.zeros([n,])
    
#验证是否所有权重都不是None    
for param_name, param_value in state_dict.items():
    if param_value is None:
        print(f"Parameter {param_name} has a value of None.")

torch.save(state_dict,"./model_int8_quant.pth")

model_int8.load_state_dict(state_dict)
model = model_int8
model.eval()

print("Load ./model_int8_quant.pth done!")
onnx_path = './yolop-640-640-quant.onnx'

input.to('cpu')
model.to('cpu')


a,b,c = model(input)
_, da_seg_mask = torch.max(b, 1)
import pdb;pdb.set_trace()

print(f"Converting to {onnx_path}")
torch.onnx.export(model, input, onnx_path,
                verbose=False, opset_version=11, input_names=['images'],
                output_names=['det_out', 'drive_area_seg', 'lane_line_seg'])
print('convert', onnx_path, 'to onnx finish!!!')











