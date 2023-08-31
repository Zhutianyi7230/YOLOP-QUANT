Perform Static quantization for the yolop model.

You can find the official source code for yolop at https://github.com/hustvl/YOLOP.

### Get Started
1. To make the quantized model.
First you should specify the path of pictures used for caliborating in quantize.py in line 26,27.
```
images_folder_bdd = '/media/zhutianyi/KESU/datasets/bdd100k/images/100k/val'
images_folder_kitti = '/media/zhutianyi/KESU/datasets/kitti/testing/image_2'
```
Then
```
python quantize.py
```

2. To use the quantized model to do inference.
```
python tools/demo_quant.py --weights /path/to/weight --source /path/to/images --device cpu
```
Results will be saved in /inference/output.
