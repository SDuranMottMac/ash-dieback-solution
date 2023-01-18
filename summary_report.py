## Package information
# ==================
__author__ = ["Sergio Duran"] # Please, include your name if you have contributed to develop this package
__License__ = "Mott MacDonald"
__maintainer__ = ["Sergio Duran"] # Please, include your name if you have contributed to develop this package
__email__ = ["sergio.duranalvarez@mottmac.com"] # Please, include your email if you have contributed to develop this package
__status__ = "Development"
__version__ = "0.0.1"

# importing libraries
# ===================
import argparse
from genericpath import isfile
import albumentations as A
import cv2
import json
import logging
from glob import glob
import numpy as np
import os
from pathlib import Path
import torch.backends.cudnn as cudnn
import shutil
from threading import Lock, Thread, Event
import threading
import _thread
import time
from time import sleep
import torch
import random
import sys
import uuid
from contextlib import contextmanager
import multiprocessing
import math
from tensorflow import keras

# importing Zed dependencies
# ===========================
sys.path.insert(1, './src/pipeline')
# from video_generation import progress_bar, turning_video_oneimage,max_length_video, new_to_avi
from deepsort_tracker import detect_and_track,reordering_deepsort_corrected,identify_smaller

# importing Yolo dependencies
# ===========================
sys.path.insert(0, './models/Object_detection/yolov5')
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.torch_utils import select_device
from utils.augmentations import letterbox

# importing Timm dependencies
# ===========================
sys.path.insert(2, './models/Image_classification/pytorch-image-models')
from get_inference import inference_classification,get_fp_assessment,health_class_Resnet50,dieback_class_Resnet50


class SummaryDieback():
    """
    This class creates a summary with the amount of ash trees and dieback level on the given videos
    """
    def __init__(self,input_folder):
        self.input_folder = input_folder
        self.yolo_weights = [r"\\gb010587mm\Software_dev\Ash_Dieback_Solution\Final_Model_weights\ash_detection\yolo_1440.pt",r"\\gb010587mm\Software_dev\Ash_Dieback_Solution\Final_Model_weights\ash_detection\yolos_1440.pt"]
        self.deepsort_model = r"\\gb010587mm\Software_dev\Ash_Dieback_Solution\ash_dieback_upgraded\weights\deepsort\osnet_ibn_x1_0_imagenet.pth"
        self.config_deepsort = r"\\gb010587mm\Software_dev\Ash_Dieback_Solution\ash_dieback_upgraded\weights\deepsort\deep_sort_15fps.yaml"
        self.dieback_model_2classes = r"\\gb010587mm\Software_dev\Ash_Dieback_Solution\Final_Model_weights\dieback_classification\Health_class_2.keras"
        self.dieback_model_4classes = r"\\gb010587mm\Software_dev\Ash_Dieback_Solution\Final_Model_weights\dieback_classification\Health_class_4.keras"
    

    def tracking_video(self,video):
        # obtaining the output path
        out_dir = os.path.dirname(video)
        # deploying yolo and tracking objects
        detect_and_track(yolo_model=self.yolo_weights,deep_sort_model=self.deepsort_model,config_deepsort=self.config_deepsort,imgsz=[1440],out=os.path.join(out_dir,"Tracked_Detections"),source=video,conf_thres=0.2,iou_thres=0.45,classes=[0,2])
        # reordering detections
        deepsort_dict=reordering_deepsort_corrected(outfile=os.path.join(out_dir,"Tracked_Detections","Tracked.txt"),max_separation=2)
        return(deepsort_dict)
    
    def tracking_to_images(self,deepsort_dict,input_video):
        for fr in deepsort_dict:
             # reading video
            video = cv2.VideoCapture(input_video)
             # extracting frame
            video.set(cv2.CAP_PROP_POS_FRAMES, int(fr))
            ret, frame = video.read()
            if ret != True:
                continue
            # obtaining ID
            for ix,id_ in enumerate(deepsort_dict[fr]["id"]):
                # 1 - create folder
                os.makedirs(os.path.join(self.input_folder,os.path.basename(input_video)[:-4],str(id_)),exist_ok=True)
                # 2 - cropping image
                x = int(deepsort_dict[fr]["bbox_left"][ix])
                y = int(deepsort_dict[fr]["bbox_top"][ix])
                w = int(deepsort_dict[fr]["bbox_w"][ix])
                h = int(deepsort_dict[fr]["bbox_h"][ix])
                cropped_image = frame[y:y+h,x:x+w]
                # 3 - image number
                n_images = len(os.listdir(os.path.join(self.input_folder,os.path.basename(input_video)[:-4],str(id_)))) + 1
                # 4 - image name
                img_name = "image_"+str(n_images)+".png"
                # 5 - Applying transformations
                transform = A.Compose([
                    A.LongestMaxSize(max_size=224,interpolation=1),
                    A.PadIfNeeded(min_height=224,min_width=224,border_mode=2)
                    ])
                transformed = transform(image=cropped_image)
                transformed_image = transformed["image"]
                # 6 - saving the image in the corresponding folder
                cv2.imwrite(os.path.join(self.input_folder,os.path.basename(input_video)[:-4],str(id_),img_name),transformed_image)
    
    def calculating_dieback(self, images_folder):
        # generating the dictionary for this video
        video_dieback = {"100-75 live crown":0,"75-50 live crown":0,"50-25 live crown":0,"25-0 live crown":0}
        id_folders = glob(images_folder+"/*/", recursive = True)
        # instantiating the models
        model2_clases = keras.models.load_model(self.dieback_model_2classes)
        model4_clases = keras.models.load_model(self.dieback_model_4classes)
        for fold in id_folders:
            # obtaining the output directories
            directory_name = os.path.dirname(fold).split("\\")[-1]
            # checking for consistency
            if len(os.listdir(os.path.join(images_folder,directory_name))) < 3:
                continue
            # assessing dieback level of all images there
            Dieback = dieback_class_Resnet50(model2_clases=model2_clases,model4_clases=model4_clases,data_folder=os.path.join(images_folder,directory_name))
            # Dieback = health_class_Resnet50(model2classes_path=self.dieback_model_2classes,model4classes_path=self.dieback_model_4classes,data_folder=os.path.join(images_folder,directory_name))
            avg_dieback = round(np.mean(Dieback),0)
            # updating the dictionary
            if int(avg_dieback) == 0:
                video_dieback["100-75 live crown"] = video_dieback["100-75 live crown"]+1
            elif int(avg_dieback) == 1:
                video_dieback["75-50 live crown"] = video_dieback["75-50 live crown"]+1
            elif int(avg_dieback) == 2:
                video_dieback["50-25 live crown"] = video_dieback["50-25 live crown"]+1
            else:
                video_dieback["25-0 live crown"] = video_dieback["25-0 live crown"]+1
        
        return(video_dieback)

    def run_summary(self):
        # generating the global dictionary
        summary_dieback = {"100-75 live crown":0,"75-50 live crown":0,"50-25 live crown":0,"25-0 live crown":0}
        files_to_remove = []
        # obtaining .avi videos
        # for root, dirs, files in os.walk(r"\\gb002339ab\One_Touch\GB010587MM\Glasgow 22\Glasgow_Driven_Survey", topdown=True):
        #     for name_ in files:
        #         if name_.endswith(".svo") and (name_.split("_")[1]=="front"):
        #             if os.path.isfile(os.path.join(self.input_folder,name_[:-4]+".avi")):
        #                 print("Video already created - skipping")
        #                 continue
        #             try:
        #                 new_to_avi(input_vid=os.path.join(root,name_),output_video=os.path.join(self.input_folder,name_[:-4]+".avi"),vid_resolution="2K")
        #             except:
        #                 print("Invalid SVO - skipped")
        # # counting videos in directory
        # n_videos = len(os.listdir(self.input_folder))
        n_svo_videos = 0
        # obtaining .avi videos
        for root, dirs, files in os.walk(r"\\gb002339ab\One_Touch\GB010587MM\Glasgow 22\Glasgow_Driven_Survey", topdown=True):
            for name_ in files:
                if name_.endswith(".svo") and (name_.split("_")[1]=="front"):
                    n_svo_videos = n_svo_videos + 1
        count = 0
        # obtaining tracking file
        for item in os.listdir(self.input_folder):
            if item.endswith(".avi") and (item.split("_")[1]=="front"):
                print("dealing with video: "+str(item))
                # obtaining tracking file
                try: 
                    deepsort_dict = self.tracking_video(os.path.join(self.input_folder,item))
                    # transforming them into detection images
                    os.makedirs(os.path.join(self.input_folder,item[:-4]),exist_ok=True)
                    self.tracking_to_images(deepsort_dict,os.path.join(self.input_folder,item))
                    # calculating dieback on the video
                    video_dieback = self.calculating_dieback(images_folder=os.path.join(self.input_folder,item[:-4]))
                    shutil.rmtree(os.path.join(self.input_folder,item[:-4]))
                    # updating the global dieback dict
                    new_100 = int(summary_dieback["100-75 live crown"] + video_dieback["100-75 live crown"])
                    new_75 = int(summary_dieback["75-50 live crown"] + video_dieback["75-50 live crown"])
                    new_50 = int(summary_dieback["50-25 live crown"] + video_dieback["50-25 live crown"])
                    new_25 = int(summary_dieback["25-0 live crown"] + video_dieback["25-0 live crown"])
                    summary_dieback.update({"100-75 live crown":new_100,"75-50 live crown":new_75,"50-25 live crown":new_50,"25-0 live crown":new_25})
                    count = count + 1
                    print("The video is: "+str(item))
                    print("The summary report after adding the video dieback is: ")
                    print(summary_dieback)
                    print("Number of videos processed: "+str(count))
                    print("The projected summary report is: ")
                    print({"100-75 live crown":str(new_100*n_svo_videos/count),"75-50 live crown":str(new_75*n_svo_videos/count),"50-25 live crown":str(new_50*n_svo_videos/count),"25-0 live crown":str(new_25*n_svo_videos/count)})

                    # removing the tracked_detections folder
                    shutil.rmtree(os.path.join(self.input_folder,"Tracked_Detections"))
                    
                except:
                    count = count + 1
                    print("No detections")
                    if os.path.isdir(os.path.join(self.input_folder,"Tracked_Detections")):
                        shutil.rmtree(os.path.join(self.input_folder,"Tracked_Detections"))
                    
        # saving the output
        print("The final summary dieback is: ")
        print(summary_dieback)
        with open(os.path.join(self.input_folder,"summary_report.json"), "w") as outfile:
            json.dump(summary_dieback, outfile)
        
        print("Finished!")
    

if __name__ == "__main__":
    """
    In order to run this code:
    1.- Define the yolo and resnet models you want to use (init method)
    2.- Create a folder with all the .svo videos you want to get the estimate from
    3.- Define such folder as input folder
    4.- Run this code
    """
    input_folder = r"\\gb010587mm\Ash_Dieback_Delivery\2022\Glasgow\avivideos2"
    test1 = SummaryDieback(input_folder = input_folder)
    test1.run_summary()

