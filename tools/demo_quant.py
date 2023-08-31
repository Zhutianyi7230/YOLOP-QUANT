import argparse
import os, sys
import shutil
import time
from pathlib import Path
import imageio

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

print(sys.path)
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import scipy.special
import numpy as np
import torchvision.transforms as transforms
import PIL.Image as image
from quantize import evaluate
from lib.models.common2 import Conv, SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect, SharpenConv
from torch.nn import Upsample
from lib.config import cfg
from lib.config import update_config
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net
from lib.dataset import LoadImages, LoadStreams
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box,show_seg_result
from lib.core.function import AverageMeter
from lib.core.postprocess import morphological_process, connect_lane
from tqdm import tqdm

# The lane line and the driving area segment branches without share information with each other and without link
YOLOP = [
    [24, 33, 42],  # Det_out_idx, Da_Segout_idx, LL_Segout_idx
    [-1, Focus, [3, 32, 3]],  # 0
    [-1, Conv, [32, 64, 3, 2]],  # 1
    [-1, BottleneckCSP, [64, 64, 1]],  # 2
    [-1, Conv, [64, 128, 3, 2]],  # 3
    [-1, BottleneckCSP, [128, 128, 3]],  # 4
    [-1, Conv, [128, 256, 3, 2]],  # 5
    [-1, BottleneckCSP, [256, 256, 3]],  # 6
    [-1, Conv, [256, 512, 3, 2]],  # 7
    [-1, SPP, [512, 512, [5, 9, 13]]],  # 8 SPP
    [-1, BottleneckCSP, [512, 512, 1, False]],  # 9
    [-1, Conv, [512, 256, 1, 1]],  # 10
    [-1, Upsample, [None, 2, 'nearest']],  # 11
    [[-1, 6], Concat, [1]],  # 12
    [-1, BottleneckCSP, [512, 256, 1, False]],  # 13
    [-1, Conv, [256, 128, 1, 1]],  # 14
    [-1, Upsample, [None, 2, 'nearest']],  # 15
    [[-1, 4], Concat, [1]],  # 16         #Encoder

    [-1, BottleneckCSP, [256, 128, 1, False]],  # 17
    [-1, Conv, [128, 128, 3, 2]],  # 18
    [[-1, 14], Concat, [1]],  # 19
    [-1, BottleneckCSP, [256, 256, 1, False]],  # 20
    [-1, Conv, [256, 256, 3, 2]],  # 21
    [[-1, 10], Concat, [1]],  # 22
    [-1, BottleneckCSP, [512, 512, 1, False]],  # 23
    [[17, 20, 23], Detect,
     [1, [[3, 9, 5, 11, 4, 20], [7, 18, 6, 39, 12, 31], [19, 50, 38, 81, 68, 157]], [128, 256, 512]]],
    # Detection head 24: from_(features from specific layers), block, nc(num_classes) anchors ch(channels)

    [16, Conv, [256, 128, 3, 1]],  # 25
    [-1, Upsample, [None, 2, 'nearest']],  # 26
    [-1, BottleneckCSP, [128, 64, 1, False]],  # 27
    [-1, Conv, [64, 32, 3, 1]],  # 28
    [-1, Upsample, [None, 2, 'nearest']],  # 29
    [-1, Conv, [32, 16, 3, 1]],  # 30
    [-1, BottleneckCSP, [16, 8, 1, False]],  # 31
    [-1, Upsample, [None, 2, 'nearest']],  # 32
    [-1, Conv, [8, 2, 3, 1]],  # 33 Driving area segmentation head

    [16, Conv, [256, 128, 3, 1]],  # 34
    [-1, Upsample, [None, 2, 'nearest']],  # 35
    [-1, BottleneckCSP, [128, 64, 1, False]],  # 36
    [-1, Conv, [64, 32, 3, 1]],  # 37
    [-1, Upsample, [None, 2, 'nearest']],  # 38
    [-1, Conv, [32, 16, 3, 1]],  # 39
    [-1, BottleneckCSP, [16, 8, 1, False]],  # 40
    [-1, Upsample, [None, 2, 'nearest']],  # 41
    [-1, Conv, [8, 2, 3, 1]]  # 42 Lane line segmentation head
]

normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])


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

def detect(cfg,opt):
    images_folder_bdd = '/media/zhutianyi/KESU/datasets/bdd100k/images/100k/val'
    images_folder_kitti = '/media/zhutianyi/KESU/datasets/kitti/testing/image_2'
    logger, _, _ = create_logger(
        cfg, cfg.LOG_DIR, 'demo')
    device = select_device(logger,opt.device)
    if os.path.exists(opt.save_dir):  # output dir
        shutil.rmtree(opt.save_dir)  # delete dir
    os.makedirs(opt.save_dir)  # make new dir
    
    # Load model
    ###
    model_fp32 = get_net(cfg)
    model_fp32.eval()
    checkpoint = torch.load("./weights/model_float32.pth", map_location= device)
    model_fp32.load_state_dict(checkpoint)

    ##fuse
    for module_name, module in model_fp32.named_children():
        for layer_index,module in module.named_children():
            if YOLOP[int(layer_index)+1][1] is Conv:
                torch.quantization.fuse_modules(module,[["conv", "bn"]],inplace=True)
            elif YOLOP[int(layer_index)+1][1] is Focus:
                for module_name,module in module.named_children():
                    if module_name=='conv':
                        torch.quantization.fuse_modules(module,[["conv", "bn"]],inplace=True)
            elif YOLOP[int(layer_index)+1][1] is SPP:
                for module_name,module in module.named_children():
                    if module_name=='cv1' or module_name=='cv2':
                        torch.quantization.fuse_modules(module,[["conv", "bn"]],inplace=True)
            elif YOLOP[int(layer_index)+1][1] is BottleneckCSP:
                for module_name,module in module.named_children():
                    if module_name=='cv1' or module_name=='cv4':
                        torch.quantization.fuse_modules(module,[["conv", "bn"]],inplace=True)
                    elif module_name=='m':
                        for module_name,module in module.named_children():
                            for module_name,module in module.named_children():
                                if module_name=='cv1' or module_name=='cv2':
                                    torch.quantization.fuse_modules(module,[["conv", "bn"]],inplace=True)
    ###

    model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model_fp32_prepared = torch.quantization.prepare(model_fp32)
    evaluate(model_fp32_prepared,images_folder_bdd,images_folder_kitti)
    model_int8 = torch.quantization.convert(model_fp32_prepared)
    model_int8.eval()
    model = model_int8
    model.eval()
    # Set Dataloader
    if opt.source.isnumeric():
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(opt.source, img_size=opt.img_size)
    else:
        dataset = LoadImages(opt.source, img_size=opt.img_size)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    inf_time = AverageMeter()
    nms_time = AverageMeter()
    yolop_time = AverageMeter()
    t99 = time_synchronized()
    
    for i, (path, img, img_det, vid_cap,shapes) in tqdm(enumerate(dataset),total = len(dataset)):
        img = transform(img).to(device)
        img =img.float()  # uint8 to fp16/32
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # # Inference
        t1 = time_synchronized()   
        det_out, da_seg_out,ll_seg_out= model(img)
        t2 = time_synchronized()

        yolop_time.update(time_synchronized()-t99,img.size(0))
        t99 = time_synchronized()
        
        inf_out,_ = det_out
        inf_time.update(t2-t1,img.size(0))

        # Apply NMS
        t3 = time_synchronized()
        det_pred = non_max_suppression(inf_out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, classes=None, agnostic=False)
        t4 = time_synchronized()

        nms_time.update(t4-t3,img.size(0))
        det=det_pred[0]

        save_path = str(opt.save_dir +'/'+ Path(path).name) if dataset.mode != 'stream' else str(opt.save_dir + '/' + "web.mp4")

        _, _, height, width = img.shape
        h,w,_=img_det.shape
        pad_w, pad_h = shapes[1][1]
        pad_w = int(pad_w)
        # pad_h = int(pad_h)
        ratio = shapes[1][0][1]
        
        da_predict = da_seg_out[:, :, int(pad_h):(height-int(pad_h+0.5)),pad_w:(width-pad_w)]
        # da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1/ratio), mode='bilinear')
        da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=(1/ratio), mode='bilinear')
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
        # da_seg_mask = morphological_process(da_seg_mask, kernel_size=7)
        ll_predict = ll_seg_out[:, :,int(pad_h):(height-int(pad_h+0.5)),pad_w:(width-pad_w)]
        # ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1/ratio), mode='bilinear')
        ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=(1/ratio), mode='bilinear')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
        # Lane line post-processing
        #ll_seg_mask = morphological_process(ll_seg_mask, kernel_size=7, func_type=cv2.MORPH_OPEN)
        #ll_seg_mask = connect_lane(ll_seg_mask)

        img_det = np.resize(img_det,(ll_seg_mask.shape[0],ll_seg_mask.shape[1],3))######
        img_det = show_seg_result(img_det, (da_seg_mask, ll_seg_mask), _, _, is_demo=True)

        if len(det):
            det[:,:4] = scale_coords(img.shape[2:],det[:,:4],img_det.shape).round()
            for *xyxy,conf,cls in reversed(det):
                label_det_pred = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img_det , label=label_det_pred, color=colors[int(cls)], line_thickness=2)
        
        if dataset.mode == 'images':
            cv2.imwrite(save_path,img_det)

        elif dataset.mode == 'video':
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer

                fourcc = 'mp4v'  # output video codec
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                h,w,_=img_det.shape
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            vid_writer.write(img_det)
        
        else:
            cv2.imshow('image', img_det)
            cv2.waitKey(1)  # 1 millisecond

    print('Results saved to %s' % Path(opt.save_dir))
    print('Done. (%.3fs)' % (time.time() - t0))
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)  yolop : (%.4fs/frame)'  % (inf_time.avg,nms_time.avg,yolop_time.avg))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./weights/model_int8_quant.pth', help='model.pth path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder   ex:inference/images
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    with torch.no_grad():
        detect(cfg,opt)


#python tools/demo.py  
