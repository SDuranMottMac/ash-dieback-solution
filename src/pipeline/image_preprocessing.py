'''
This file contains a series of functions that intend to deal with image
pre-processing, post-processing and manipulation
'''

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.torch_utils import select_device
from utils.augmentations import letterbox
import numpy as np
import torch
import cv2
import pyzed.sl as sl

def img_preprocess(img, device, half, net_size):
    """
    This function preprocess images to be consumed by Yolov5
    """
    # adapting the image
    net_image, ratio, pad = letterbox(img[:, :, :3], net_size, auto=False)
    # transposing image
    net_image = net_image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    net_image = np.ascontiguousarray(net_image)
    # loading image into gpu
    img = torch.from_numpy(net_image).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    # normalising image
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    # creating a numpy ndarray with a 4th dimension
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img, ratio, pad

def xywh2abcd(xywh, im_shape):
    """
    This function turns xywh bounding box coordinates into ABCD coordinates
    """
    
    output = np.zeros((4, 2))

    # Center / Width / Height -> BBox corners coordinates
    x_min = (xywh[0] - 0.5*xywh[2]) * im_shape[1]
    x_max = (xywh[0] + 0.5*xywh[2]) * im_shape[1]
    y_min = (xywh[1] - 0.5*xywh[3]) * im_shape[0]
    y_max = (xywh[1] + 0.5*xywh[3]) * im_shape[0]

    # A ------ B
    # | Object |
    # D ------ C

    # Point A
    output[0][0] = x_min
    output[0][1] = y_min
    # Point B
    output[1][0] = x_max
    output[1][1] = y_min
    # Point C
    output[2][0] = x_min
    output[2][1] = y_max
    # Point D
    output[3][0] = x_max
    output[3][1] = y_max
    return(output)

def bounding_boxes(input_box):

    output_box = ((input_box[0][0],input_box[0][1]),(input_box[2][0],input_box[2][1]))
    
    return(output_box)

def get_center_box(input_box):

    center_x = int((int(input_box[1][0]) - int(input_box[0][0])) / 2)
    center_y = int((int(input_box[1][1]) - int(input_box[2][1])) / 2)
    
    return(center_x,center_y)

def drawing_bbox(img,bbox,outimg):
    """
    This function draws bounding boxes on plain images
    """

    # loading image
    image = cv2.imread(img)
    # defining bounding box
    starting_point = (int(bbox[0][0]),int(bbox[0][1]))
    end_point = (int(bbox[1][0]),int(bbox[1][1]))
    # color and font
    color_ = (0,255,255)
    font_color = (0,128,128)
    thickness = 3
    # drawing the rectangle
    image = cv2.rectangle(image, starting_point, end_point, color_, thickness)
    # drawing label
    start_rect = (int(bbox[0][0]),int(bbox[0][1]))
    fin_point = (int(bbox[0][0] + 180),int(bbox[0][1] + 40))
    image = cv2.rectangle(image, start_rect, fin_point, color_, -1)
    image = cv2.putText(image, "Detection", (int(bbox[0][0]+20),int(bbox[0][1]+35)),cv2.FONT_HERSHEY_SIMPLEX,1,font_color,2,cv2.LINE_AA)
    cv2.imwrite(outimg, image) 

def drawing_bbox_and_distance(img,bbox,outimg,distance):
    """
    This function draws bounding boxes and the estimated distance on plain images
    """

    # loading image
    image = cv2.imread(img)
    # defining bounding box
    starting_point = (int(bbox[0][0]),int(bbox[0][1]))
    end_point = (int(bbox[1][0]),int(bbox[1][1]))
    # color and font
    color_ = (0,0,255)
    font_color = (0,0,0)
    thickness = 3
    # drawing the rectangle
    image = cv2.rectangle(image, starting_point, end_point, color_, thickness)
    # drawing label
    start_rect = (int(bbox[0][0]),int(bbox[0][1]))
    fin_point = (int(bbox[0][0] + 180),int(bbox[0][1] + 40))
    image = cv2.rectangle(image, start_rect, fin_point, color_, -1)
    image = cv2.putText(image, "Ash Tree", (int(bbox[0][0]+20),int(bbox[0][1]+35)),cv2.FONT_HERSHEY_SIMPLEX,1,font_color,2,cv2.LINE_AA)
    image = cv2.putText(image, "Distance: "+str(distance), (int(bbox[0][0]+20),int(bbox[0][1]-35)),cv2.FONT_HERSHEY_SIMPLEX,1,font_color,2,cv2.LINE_AA)
    cv2.imwrite(outimg, image)

def new_detections_to_custom_box(detections, im, im0):
    """
    This function transform yolo detections into custom Zed box data
    """
    output = []
    for i, det in enumerate(detections):
        if len(det):
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                # Creating ingestable objects for the ZED SDK
                obj = sl.CustomBoxObjectData()
                obj.unique_object_id = sl.generate_unique_id()
                obj.bounding_box_2d = xywh2abcd(xywh, im0.shape)
                obj.label = cls
                obj.probability = conf
                obj.is_grounded = True
                output.append(obj)
    return output 

def detections_to_custom_box(detections, im, im0):


    output = []
    for i, det in enumerate(detections):
        if len(det):
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                # Creating ingestable objects for the ZED SDK
                obj = sl.CustomBoxObjectData()
                obj.bounding_box_2d = xywh2abcd(xywh, im0.shape)
                obj.label = cls
                obj.probability = conf
                obj.is_grounded = False
                output.append(obj)
    return output 

def generate_YOLO_label(bbox,class_number,img_shape,score):
    """
    Generate yolo label rows to fill txt labels
    """
    #((input_box[0][0],input_box[0][1]),(input_box[2][0],input_box[2][1]))
    bounding_box = bounding_boxes(input_box=bbox)
    x_c,y_c = get_center_box(input_box=bbox)
    width = int(bounding_box[1][0] - bounding_box[0][0])
    height = int(bounding_box[1][1] - bounding_box[0][1])

    label = str(class_number)+" "+str(score)+" "+str(x_c/img_shape[0])+" "+str(y_c/img_shape[1])+" "+str(width/img_shape[0])+" "+str(height/img_shape[1])+"\n"

    return(label)






