import cv2
import torch
from torchvision import transforms
from PIL import Image
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
import numpy as np

# 定义红绿灯阈值分割函数
def detect_traffic_light(image_path):
    # 读取图像
    image = cv2.imread(image_path)

    # 将图像转换为HSV颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义红色和绿色的HSV阈值范围
    lower_red1 = np.array([0, 43, 46])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([156, 43, 46])
    upper_red2 = np.array([180, 255, 255])
    #lower_yellow = np.array([26, 43, 46])
    #upper_yellow = np.array([34, 255, 255])
    lower_green = np.array([35, 43, 46])
    upper_green = np.array([77, 255, 255])

    # 对图像进行阈值分割，提取红色和绿色区域
    red_mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    #yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # 对红色和绿色区域进行形态学操作，去除噪声
    kernel = np.ones((5, 5), np.uint8)
    red_mask1 = cv2.morphologyEx(red_mask1, cv2.MORPH_OPEN, kernel)
    red_mask2 = cv2.morphologyEx(red_mask2, cv2.MORPH_OPEN, kernel)
    #yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

    # 计算红色和绿色区域的像素数量
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



# 加载YOLOv5模型
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# model = attempt_load("yolov5s.pt", map_location=torch.device('cpu'))
# model.eval()

# # 设置图像预处理的变换
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

while True:
    # 读取一帧图像
    image_path = r"D:\vscodepython\QCar_YOLO\red.jpg"
    ret, frame = cv2.read(image_path)

    # 将图像转换为PIL图像
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 图像预处理
    input_tensor = transform(pil_image)

    #model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


    #加载yolov5模型,需要替换成yolov5s.pt权重的地址
    model = torch.hub.load('./yolov5', 'custom', path='D:\vscodepython\QCar_YOLO\yolov5s.pt',source='local')
    model.conf = 0.4
    model = attempt_load("yolov5s.pt", map_location=torch.device('cpu'))
    checkpoint = torch.load('yolov5s.pt', map_location=torch.device('cpu'))  
    model.load_state_dict(checkpoint['model'])

    #设置模型为推理模式
    model.eval()

    # 设置图像预处理的变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 执行YOLOv5检测非时序预测模型
    with torch.no_grad():
        detections = model.forward(input_tensor.unsqueeze(0))

    # 应用非最大抑制，获取检测结果
    detections = non_max_suppression(detections, conf_thres=0.4, iou_thres=0.3)

    # 遍历每个检测结果
    for detection in detections[0]:
        label = detection[-1]  # 获取类别标签
        bbox = detection[:4] * torch.Tensor([pil_image.width, pil_image.height, pil_image.width, pil_image.height])

        # 计算检测框的中心坐标
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2

        # 调用图像截取函数，截取检测框所在位置的图像
        cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # 显示截取的图像，并输出类别和中心坐标
        cv2.imshow("Cropped Image", cropped_image)
        print("类别：{}, 中心坐标：({}, {})".format(label, center_x, center_y))

    if label == "traffic light":
        result = detect_traffic_light(cropped_image)
        print(result)

    # 按下q键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # 显示原始图像
    #cv2.imshow('Original Image', frame)
