# importing libraries
import argparse
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import platform
import shutil
import time
from pathlib import Path
import random
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from itertools import groupby
import json
from operator import itemgetter
import sys
from yolov5_deepsort.models.experimental import attempt_load
from yolov5_deepsort.utils.downloads import attempt_download
from yolov5_deepsort.models.common import DetectMultiBackend
from yolov5_deepsort.utils.datasets import LoadImages, LoadStreams, VID_FORMATS
from yolov5_deepsort.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr)
from yolov5_deepsort.utils.torch_utils import select_device, time_sync
from yolov5_deepsort.utils.plots import Annotator, colors, save_one_box
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import pandas as pd
import random
import cv2


def detect_and_track(yolo_model,deep_sort_model,config_deepsort,imgsz,out,source,conf_thres,iou_thres=0.45,classes=[0,2]):
    
    show_vid=False
    save_vid=False
    save_txt=True
    imgsz=[1440]
    evaluate=False
    half=False
    project = out
    exist_ok = False
    update = False
    save_crop = False

    device=''

    imgsz *= 2 if len(imgsz) == 1 else 1
    # Initialize
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Directories
    if type(yolo_model) is str:  # single yolo model
        exp_name = yolo_model.split(".")[0]
    elif type(yolo_model) is list and len(yolo_model) == 1:  # single models after --yolo_model
        exp_name = yolo_model[0].split(".")[0]
    else:  # multiple models after --yolo_model
        exp_name = "ensemble"
    exp_name = exp_name + "_" + deep_sort_model.split('/')[-1].split('.')[0]

    # Load model
    model = DetectMultiBackend(yolo_model, device=device, dnn=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Dataloader
    if False:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(config_deepsort)

    # Create as many trackers as there are video sources
    deepsort_list = []
    for i in range(nr_sources):
        deepsort_list.append(
            DeepSort(
                deep_sort_model,
                device,
                max_dist=cfg.DEEPSORT.MAX_DIST,
                max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
            )
        )
    outputs = [None] * nr_sources

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run tracking
    model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    cap = cv2.VideoCapture(source)
    length_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(tqdm(dataset,total=length_video)):
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = False
        pred = model(im, augment=False, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if False:  # nr_sources >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                #if source.endswith(VID_FORMATS):
                if source.endswith(".avi"):
                    txt_file_name = "Tracked"
                    save_path = os.path.join(out,p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = os.path.join(out,p.parent.name)  # im.jpg, vid.mp4, ...

            txt_path = os.path.join(out,txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=2, pil=not ascii)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                t4 = time_sync()
                outputs[i] = deepsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    for j, (output) in enumerate(outputs[i]):

                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        conf = output[6]

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, conf, -1, -1, i))

                        if save_vid or save_crop or show_vid:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = f'{id:0.0f} {names[c]} {conf:.2f}'
                            annotator.box_label(bboxes, label, color=colors(c, True))
                            if save_crop:
                                txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                save_one_box(bboxes, imc, file=out / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)

                #LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')

            else:
                deepsort_list[i].increment_ages()
                #LOGGER.info('No detections')

            # Stream results
            im0 = annotator.result()
            if show_vid:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)
    
def reordering_deepsort(outfile):
    deepsort_dict = dict()
    with open(outfile, 'r') as f:
        lines = f.readlines()

    for item in lines:

        detections = item.split(' ')
        if detections[0] in deepsort_dict:
            deepsort_dict[detections[0]]["id"].append(detections[1])
            deepsort_dict[detections[0]]["bbox_left"].append(detections[2])
            deepsort_dict[detections[0]]["bbox_top"].append(detections[3])
            deepsort_dict[detections[0]]["bbox_w"].append(detections[4])
            deepsort_dict[detections[0]]["bbox_h"].append(detections[5])
            deepsort_dict[detections[0]]["conf"].append(detections[6])
        else:
            #bbox_left,  # MOT format bbox_top, bbox_w, bbox_h
            deepsort_dict[detections[0]]= {"id":[detections[1]], "bbox_left":[detections[2]], "bbox_top":[detections[3]], "bbox_w":[detections[4]], "bbox_h":[detections[5]],"conf":[detections[6]]}

    # dictionary to json
    return(deepsort_dict)

def postprocessing_id(dict1):

    # creating empty dictionaries
    frames_appearance = []
    all_ids = []
    all_bbox_left = []
    all_bbox_top = []
    all_bbox_w =[]
    all_bbox_h = []
    all_conf = []

    #get all the ids, bbox, w, h, top, left
    for key in dict1:
        for id_ in dict1[key]["id"]:
            all_ids.append(id_)
            frames_appearance.append(key)
    
    for key in dict1:
        for id_ in dict1[key]["bbox_left"]:
            all_bbox_left.append(id_)

    for key in dict1:
        for id_ in dict1[key]["bbox_top"]:
            all_bbox_top.append(id_)  

    for key in dict1:
        for id_ in dict1[key]["bbox_w"]:
            all_bbox_w.append(id_) 

    for key in dict1:
        for id_ in dict1[key]["bbox_h"]:
            all_bbox_h.append(id_)   
    
    for key in dict1:
        for id_ in dict1[key]["conf"]:
            all_conf.append(id_)  

    # get the unique ids
    #unique_ids = list(set(all_ids))
    sequence_ids = dict()
    for idx,id_ in enumerate(all_ids):
        if id_ in sequence_ids:
            sequence_ids[id_]["frames"].append(frames_appearance[idx])
            sequence_ids[id_]["bbox_left"].append(all_bbox_left[idx])
            sequence_ids[id_]["bbox_top"].append(all_bbox_top[idx])
            sequence_ids[id_]["bbox_w"].append(all_bbox_w[idx])
            sequence_ids[id_]["bbox_h"].append(all_bbox_h[idx])
            sequence_ids[id_]["conf"].append(all_conf[idx])
        else:
            sequence_ids[id_] = {"frames":[frames_appearance[idx]], "bbox_left":[all_bbox_left[idx]], "bbox_top":[all_bbox_top[idx]], "bbox_w":[all_bbox_w[idx]], "bbox_h":[all_bbox_h[idx]],"conf":[all_conf[idx]]}
    
    return(sequence_ids)
    
    
def consecutive_number(list_,max_separation=2):
    """
    Returns a list of consecutive numbers from a list of integers.
    """
    #converting to integer
    list_ = [int(i) for i in list_]

    unique_ids = []
    for i,item in enumerate(list_):
        if i == 0:
            unique_ids.append([item])
        else:
            if ((item - list_[i-1])<max_separation):
                unique_ids[-1].append(item)
            else:
                unique_ids.append([item])  
    return(unique_ids)

def correcting_mixed_ids(sequence_ids,max_separation=2):
    corrected_ids = dict()
    for id_ in sequence_ids:
        unique_ids = consecutive_number(sequence_ids[id_]["frames"],max_separation=max_separation)
        if len(unique_ids)==1:
            # transform list to string
            uniqued = [str(i) for i in unique_ids[0]]
            corrected_ids[id_] = {"frames":uniqued, "bbox_left":sequence_ids[id_]["bbox_left"], "bbox_top":sequence_ids[id_]["bbox_top"], "bbox_w":sequence_ids[id_]["bbox_w"], "bbox_h":sequence_ids[id_]["bbox_h"],"conf":sequence_ids[id_]["conf"]}
        else:
            added_ids = len(unique_ids) - 1
            uniqued = [str(i) for i in unique_ids[0]]
            n_items = len(uniqued)
            corrected_ids[id_] = {"frames":uniqued, "bbox_left":sequence_ids[id_]["bbox_left"][:n_items], "bbox_top":sequence_ids[id_]["bbox_top"][:n_items], "bbox_w":sequence_ids[id_]["bbox_w"][:n_items], "bbox_h":sequence_ids[id_]["bbox_h"][:n_items],"conf":sequence_ids[id_]["conf"][:n_items]}
            for i in range(added_ids):
                new_id = str(int(id_) + i + random.randint(10000,100000))
                uniqued = [str(i) for i in unique_ids[i+1]]
                previous_n = len(unique_ids[i])
                n_items = len(uniqued)
                corrected_ids[new_id] = {"frames":uniqued, "bbox_left":sequence_ids[id_]["bbox_left"][previous_n:previous_n+n_items], "bbox_top":sequence_ids[id_]["bbox_top"][previous_n:previous_n+n_items], "bbox_w":sequence_ids[id_]["bbox_w"][previous_n:previous_n+n_items], "bbox_h":sequence_ids[id_]["bbox_h"][previous_n:previous_n+n_items],"conf":sequence_ids[id_]["conf"][previous_n:previous_n+n_items]}

    return(corrected_ids)
    
def reordering_deepsort_corrected(outfile,max_separation=2):
    deepsort_dict = reordering_deepsort(outfile)
    sequence_ids=postprocessing_id(dict1=deepsort_dict)
    corrected_id=correcting_mixed_ids(sequence_ids=sequence_ids,max_separation=max_separation)

    #first, removing stuff that is not there
    for key in deepsort_dict:
        index_to_remove = []
        for k,id_ in enumerate(deepsort_dict[key]["id"]):
            if key not in corrected_id[id_]["frames"]:
                index_to_remove.append(k)
        
        for index in sorted(index_to_remove, reverse=True):
            del deepsort_dict[key]["id"][index]
            del deepsort_dict[key]["bbox_left"][index]
            del deepsort_dict[key]["bbox_top"][index]
            del deepsort_dict[key]["bbox_w"][index]
            del deepsort_dict[key]["bbox_h"][index]
            del deepsort_dict[key]["conf"][index]

    # second, adding new stuff
    # selecting which ids have been added
    all_ids = []
    added_ids = [] 
    for key in deepsort_dict:
        for id_ in deepsort_dict[key]["id"]:
            all_ids.append(id_)
    for key in corrected_id:
        if key not in all_ids:
            added_ids.append(key)
    
    for key in added_ids:
        for idx,frame in enumerate(corrected_id[key]["frames"]):
            deepsort_dict[frame]["id"].append(key)
            deepsort_dict[frame]["bbox_left"].append(corrected_id[key]["bbox_left"][idx])
            deepsort_dict[frame]["bbox_top"].append(corrected_id[key]["bbox_top"][idx])
            deepsort_dict[frame]["bbox_w"].append(corrected_id[key]["bbox_w"][idx])
            deepsort_dict[frame]["bbox_h"].append(corrected_id[key]["bbox_h"][idx])
            deepsort_dict[frame]["conf"].append(corrected_id[key]["conf"][idx])

    # we need to add all the bbox as well in the previous steps to be able to solve it
    return(deepsort_dict)

def identify_smaller(dist_list):
    min_value = min(dist_list)
    min_index = dist_list.index(min_value)
    return(min_index)
    

def video_to_images(input_vid,out_images):
    yolo_model=r"\\gb010587mm\Software_dev\Ash_Dieback_Solution\ash_dieback_solution\weights\yolov5\best.pt"
    deep_sort_model=r'\\gb010587mm\Software_dev\Ash_Dieback_Solution\ash_dieback_solution\weights/deepsort/osnet_x1_0_imagenet.pth'
    config_deepsort=r'\\gb010587mm\Software_dev\Ash_Dieback_Solution\ash_dieback_solution\weights/deepsort/deep_sort_15fps.yaml'

    detect_and_track(yolo_model=yolo_model,deep_sort_model=deep_sort_model,config_deepsort=config_deepsort,imgsz=[1280],out=os.path.join(r"C:\Users\DUR94074\OneDrive - Mott MacDonald\Desktop\out_things","Tracked_Detections"),source=input_vid,conf_thres=0.2,iou_thres=0.45,classes=[0,2])
    deepsort_dict = reordering_deepsort_corrected(outfile=r"C:\Users\DUR94074\OneDrive - Mott MacDonald\Desktop\out_things\Tracked_Detections\Tracked.txt",max_separation=3)

    # now we need to convert the deepsort_dic to a json
    vidcap = cv2.VideoCapture(input_vid)
    success,image = vidcap.read()
    count = 0
    while success:

        #write rectangle on image
        if str(count) in deepsort_dict:
            for indx,det in enumerate(deepsort_dict[str(count)]["id"]):
                #cv2.rectangle(image,(int(float(deepsort_dict[str(count)]["bbox_left"][indx])-float(deepsort_dict[str(count)]["bbox_w"][indx])/2),int(float(deepsort_dict[str(count)]["bbox_top"][indx])-float(deepsort_dict[str(count)]["bbox_h"][indx])/2)),(int(float(deepsort_dict[str(count)]["bbox_left"][indx])+float(deepsort_dict[str(count)]["bbox_w"][indx])/2),int(float(deepsort_dict[str(count)]["bbox_top"][indx])+float(deepsort_dict[str(count)]["bbox_h"][indx])/2)),(0,255,0),2)
                cv2.rectangle(image,(int(float(deepsort_dict[str(count)]["bbox_left"][indx])),int(float(deepsort_dict[str(count)]["bbox_top"][indx]))),(int(float(deepsort_dict[str(count)]["bbox_left"][indx])+float(deepsort_dict[str(count)]["bbox_w"][indx])/1),int(float(deepsort_dict[str(count)]["bbox_top"][indx])+float(deepsort_dict[str(count)]["bbox_h"][indx])/1)),(0,255,0),2)
                cv2.putText(image,det,(int(float(deepsort_dict[str(count)]["bbox_left"][indx])+50),int(float(deepsort_dict[str(count)]["bbox_top"][indx])+50)),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4)

        cv2.imwrite(os.path.join(out_images,"frame%d.jpg" % count), image) 
        
        # save frame as JPEG file      
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1

if __name__ == "__main__":
    
    in_folder = r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\Zed2i\training_sets\N_Yorkshire\NYorkshire_2\ls_videos"
    out_folder = r"\\gb010587mm\Software_dev\Ash_Dieback_Solution\Tests\Monmouth_val\Videos"
    vid = r"\\gb010587mm\Software_dev\Ash_Dieback_Solution\Tests\Monmouth_val\monmouth_val.avi"

    
    detect_and_track(yolo_model=r"\\gb010587mm\Software_dev\Ash_Dieback_Solution\Final_Model_weights\ash_detection\yolo_1440.pt",deep_sort_model=r"\\gb010587mm\Software_dev\Ash_Dieback_Solution\ash_dieback_upgraded\weights\deepsort\osnet_x1_0_imagenet.pth",config_deepsort=r"\\gb010587mm\Software_dev\Ash_Dieback_Solution\ash_dieback_upgraded\weights\deepsort\deep_sort_15fps.yaml",imgsz=[1440],out=out_folder,source=vid,conf_thres=0.2,iou_thres=0.45,classes=[0,2])
    # deepsort_dict=reordering_deepsort_corrected(outfile=os.path.join(out_folder,"Tracked.txt"),max_separation=3)
    # with open(os.path.join(out_folder,"deepsort_dict.json"), "w") as outfile:
    #     json.dump(deepsort_dict, outfile)

    
    
