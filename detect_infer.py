from flask import Flask, Response, render_template
import cv2
# 여기서 detect는 수정된 YOLOv5 탐지 스크립트입니다.
import torch
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.classes = [0, 1, 2, 3, 5, 7]
model.conf = 0.25
model.iou = 0.45
model.agnostic= True
model.line = 2
names = {0:'person', 1:'bicycle', 2:'car', 3:'motorcycle', 5:'bus', 7:'truck'}
device = 'cuda:0'
cap = cv2.VideoCapture(0)  # 웹캠 사용
cv2.VideoWriter_fourcc('M','J','P','G')

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def generate_frames():
    while True:
        # 비디오 프레임 읽기
        success, img = cap.read()
        if not success:
            break
        else:
            gn = torch.tensor(img.shape, device='cuda:0')[[1, 0, 1, 0]]
      
            pred = model([img]) # 예측하기
            det = pred.pred[0]
            
            if len(det):
                for i in det:
                    xywh = (xyxy2xywh(torch.tensor(i[:4]).view(1, 4))/ gn).view(-1).tolist()
                    w1 = 0.4        # w1 : 1m에서 측정한 Cart b-box width
                    h1 = 0.44       # h1 : 1m에서 측정한 Cart b-box height                               
                    std = w1 / h1
                    
                    # width값의 오차율이 height보다 작을 떄
                    if std <= xywh[2] / xywh[3]:
                        dis = round(w1 / xywh[2]*100)
                    
                    # width값의 오차율이 height보다 클 때
                    else:
                        dis = round(h1 / xywh[3]*100)

                    c = int(i[5])  # integer class
                    if dis <=100:
                        label =  f'{names[c]} {i[4]:.2f} {dis/100:.2f} Warning!'
                        colors = (0, 0, 255)
                    else :
                        label =  f'{names[c]} {i[4]:.2f} {dis/100:.2f}'
                        colors = (0, 255, 0)                        
                    plot_one_box(i[:4], img, label=label, color=colors, line_thickness=model.line)
                 

           
            cv2.imshow('frame', img)

        if cv2.waitKey(1) == ord('s'):
                break
                
    cap.release()
    cv2.destroyAllWindows()

generate_frames()