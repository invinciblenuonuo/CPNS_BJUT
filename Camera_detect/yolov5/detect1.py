import argparse
import os
import platform
import random
import sys
import cv2
from cv2 import VideoCapture
import torch
from pathlib import Path
import numpy as np
from qvl.stop_sign import QLabsStopSign
from qvl.traffic_light import QLabsTrafficLight
from qvl.qlabs import QuanserInteractiveLabs
from qvl.free_camera import QLabsFreeCamera
from qvl.basic_shape import QLabsBasicShape
from qvl.qcar import QLabsQCar
from qvl.environment_outdoors import QLabsEnvironmentOutdoors
from qvl.person import QLabsPerson
from torch.backends import cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode, time_sync



class YoloOpt:
    def __init__(self, weights='yolov5s.pt',
                 imgsz=(640, 640), conf_thres=0.25,
                 iou_thres=0.45, device='cpu', view_img=False,
                 classes=None, agnostic_nms=False,
                 augment=False, update=False, exist_ok=False,
                 project='/detect/result', name='result_exp',
                 save_csv=True):
        self.weights = weights  # 权重文件地址
        self.source = None  # 待识别的图像
        if imgsz is None:
            self.imgsz = (640, 640)
        self.imgsz = imgsz  # 输入图片的大小，默认 (640,640)
        self.conf_thres = conf_thres  # object置信度阈值 默认0.25  用在nms中
        self.iou_thres = iou_thres  # 做nms的iou阈值 默认0.45   用在nms中
        self.device = device  # 执行代码的设备，由于项目只能用 CPU，这里只封装了 CPU 的方法
        self.view_img = view_img  # 是否展示预测之后的图片或视频 默认False
        self.classes = classes  # 只保留一部分的类别，默认是全部保留
        self.agnostic_nms = agnostic_nms  # 进行NMS去除不同类别之间的框, 默认False
        self.augment = augment  # augmented inference TTA测试时增强/多尺度预测，可以提分
        self.update = update  # 如果为True,则对所有模型进行strip_optimizer操作,去除pt文件中的优化器等信息,默认为False
        self.exist_ok = exist_ok  # 如果为True,则对所有模型进行strip_optimizer操作,去除pt文件中的优化器等信息,默认为False
        self.project = project  # 保存测试日志的参数，本程序没有用到
        self.name = name  # 每次实验的名称，本程序也没有用到
        self.save_csv = save_csv  # 是否保存成 csv 文件，本程序目前也没有用到


class DetectAPI:
    def __init__(self, weights, imgsz=640):
        self.opt = YoloOpt(weights=weights, imgsz=imgsz)
        weights = self.opt.weights
        imgsz = self.opt.imgsz

        # Initialize 初始化
        # 获取设备 CPU/CUDA
        self.device = select_device(self.opt.device)
        # 不使用半精度
        self.half = self.device.type != 'cpu'  # # FP16 supported on limited backends with CUDA

        # Load model 加载模型
        self.model = DetectMultiBackend(weights, self.device, dnn=False)
        self.stride = self.model.stride
        self.names = self.model.names
        self.pt = self.model.pt
        self.imgsz = check_img_size(imgsz, s=self.stride)

        # 不使用半精度
        if self.half:
            self.model.half() # switch to FP16

        # read names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

    def detect(self, source):
        # 输入 detect([img])
        if type(source) != list:
            raise TypeError('source must a list and contain picture read by cv2')

        # DataLoader 加载数据
        # 直接从 source 加载数据
        dataset = LoadImages(source)
        # 源程序通过路径加载数据，现在 source 就是加载好的数据，因此 LoadImages 就要重写
        bs = 1 # set batch size

        # 保存的路径
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        result = []
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
                next(self.model.parameters())))  # run once
        dt, seen = (Profile(), Profile(), Profile()), 0

        for im, im0s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(self.model.device)
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

                # Inference
                pred = self.model(im, augment=self.opt.augment)[0]

                # NMS
                with dt[2]:
                    pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, self.opt.classes, self.opt.agnostic_nms, max_det=2)

                # Process predictions
                # 处理每一张图片
                det = pred[0]  # API 一次只处理一张图片，因此不需要 for 循环
                im0 = im0s.copy()  # copy 一个原图片的副本图片
                result_txt = []  # 储存检测结果，每新检测出一个物品，长度就加一。
                                 # 每一个元素是列表形式，储存着 类别，坐标，置信度
                # 设置图片上绘制框的粗细，类别名称
                annotator = Annotator(im0, line_width=3, example=str(self.names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    # 映射预测信息到原图
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    #
                    for *xyxy, conf, cls in reversed(det):
                        line = (int(cls.item()), [int(_.item()) for _ in xyxy], conf.item())  # label format
                        result_txt.append(line)
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        annotator.box_label(xyxy, label, color=self.colors[int(cls)])
                result.append((im0, result_txt))  # 对于每张图片，返回画完框的图片，以及该图片的标签列表。
            return result, self.names

def detect_traffic_light(image):
    # 读取图像
    image = cv2.imread(image)

    # 将图像转换为HSV颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义红色、黄色和绿色的HSV阈值范围
    lower_red1 = np.array([0, 43, 46])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([156, 43, 46])
    upper_red2 = np.array([180, 255, 255])
    #lower_yellow = np.array([26, 43, 46])
    #upper_yellow = np.array([34, 255, 255])
    lower_green = np.array([35, 43, 46])
    upper_green = np.array([77, 255, 255])

    # 对图像进行阈值分割，提取红色、黄色和绿色区域
    red_mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    #yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # 对红色、黄色和绿色区域进行形态学操作，去除噪声
    kernel = np.ones((5, 5), np.uint8)
    red_mask1 = cv2.morphologyEx(red_mask1, cv2.MORPH_OPEN, kernel)
    red_mask2 = cv2.morphologyEx(red_mask2, cv2.MORPH_OPEN, kernel)
    #yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

    # 计算红色、黄色和绿色区域的像素数量
    red_pixels = cv2.countNonZero(red_mask1) + cv2.countNonZero(red_mask2)
    #yellow_pixels = cv2.countNonZero(yellow_mask)
    green_pixels = cv2.countNonZero(green_mask)
    # 判断红色、黄色和绿色区域的像素数量，确定颜色
    if red_pixels > green_pixels:
        return "red"
    
    elif green_pixels > red_pixels:
        return "green"
    else:
        return "未知"
    
def main():

    image = cv2.VideoCapture(0)

    # ret, frame = image.read()
    # detect_traffic_light(image)

while True:
    # 读取视频流的帧
    image = cv2.VideoCapture(0)
    ret, frame = image.read()
    if not ret:
        break

    # 调整图像大小，加快处理速度
    frame = cv2.resize(frame, (640, 480))

    # 检测红绿灯颜色
    result = detect_traffic_light(frame)

    # 在帧上绘制识别结果
    cv2.putText(frame, result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Traffic_Light_Detection', frame)
    
    # 按 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# #释放视频流和窗口
    VideoCapture.release()
    cv2.destroyAllWindows()


# if __name__ == '__main__':
#     cap = cv2.VideoCapture(0)
#     a = DetectAPI(weights='yolov5s.pt')
#     with torch.no_grad():
#         while True:
#             rec, img = cap.read()
#             result, names = a.detect([img])
#             img = result[0][0]  # 每一帧图片的处理结果图片
#             # 每一帧图像的识别结果（可包含多个物体）
#             for cls, (x1, y1, x2, y2), conf in result[0][1]:
#                 print(names[cls], x1, y1, x2, y2, conf) 
#             print()  # 将每一帧的结果输出分开
#             cv2.imshow("video", img)

#             if cv2.waitKey(1) == ord('q'):
#                 break

main()