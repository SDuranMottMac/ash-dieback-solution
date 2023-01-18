# importing libraries
import os
import shutil
from pathlib import Path
import random
import cv2
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import sys
sys.path.insert(0, './models/Object_detection/yolov5')
sys.path.insert(1, './src/pipeline')
import pandas as pd
import random
import math
from PIL import ImageStat, Image
from yolov5_deepsort.models.common import DetectMultiBackend
from yolov5_deepsort.utils.datasets import LoadImages, LoadStreams, VID_FORMATS
from yolov5_deepsort.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords,
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5_deepsort.utils.torch_utils import select_device, time_sync
from yolov5_deepsort.utils.plots import Annotator, colors, save_one_box
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from video_generation import new_to_avi as avi_video

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def detect_and_track_conf(yolo_model,deep_sort_model,config_deepsort,imgsz,out,source,conf_thres,iou_thres=0.45,classes=[0,2]):
    
    show_vid=False
    save_vid=False
    save_txt=True
    imgsz=[640]
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
            pass  # delete output folder
        os.makedirs(out,exist_ok=True)  # make new output folder

    # Directories
    if type(yolo_model) is str:  # single yolo model
        exp_name = yolo_model.split(".")[0]
    elif type(yolo_model) is list and len(yolo_model) == 1:  # single models after --yolo_model
        exp_name = yolo_model[0].split(".")[0]
    else:  # multiple models after --yolo_model
        exp_name = "ensemble"
    exp_name = exp_name + "_" + deep_sort_model.split('/')[-1].split('.')[0]
    # save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run if project name exists
    # (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

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

def reordering_deepsort_conf(outfile):
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

def brightness(im_file):

    img = Image.fromarray(im_file)
    stat = ImageStat.Stat(img)
    r,g,b = stat.mean

    return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))

def postprocessing_id_conf(dict1):

    # creating empty dictionaries
    frames_appearance = []
    all_ids = []
    all_bbox_left = []
    all_bbox_top = []
    all_bbox_w =[]
    all_bbox_h = []
    all_conf = []

    #get all the ids, bbox, w, h, top, left, conf
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
            sequence_ids[id_] = {"frames":[frames_appearance[idx]], "bbox_left":[all_bbox_left[idx]], "bbox_top":[all_bbox_top[idx]], "bbox_w":[all_bbox_w[idx]], "bbox_h":[all_bbox_h[idx]], "conf":[all_conf[idx]]}
    
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

def correcting_mixed_ids_conf(sequence_ids,max_separation=2):
    corrected_ids = dict()
    for id_ in sequence_ids:
        try:
            unique_ids = consecutive_number(sequence_ids[id_]["frames"],max_separation=max_separation)
            if len(unique_ids)==1:
                # transform list to string
                uniqued = [str(i) for i in unique_ids[0]]
                corrected_ids[id_] = {"frames":uniqued, "bbox_left":sequence_ids[id_]["bbox_left"], "bbox_top":sequence_ids[id_]["bbox_top"], "bbox_w":sequence_ids[id_]["bbox_w"], "bbox_h":sequence_ids[id_]["bbox_h"], "conf":sequence_ids[id_]["conf"]}
            else:
                added_ids = len(unique_ids) - 1
                uniqued = [str(i) for i in unique_ids[0]]
                n_items = len(uniqued)
                corrected_ids[id_] = {"frames":uniqued, "bbox_left":sequence_ids[id_]["bbox_left"][:n_items], "bbox_top":sequence_ids[id_]["bbox_top"][:n_items], "bbox_w":sequence_ids[id_]["bbox_w"][:n_items], "bbox_h":sequence_ids[id_]["bbox_h"][:n_items], "conf":sequence_ids[id_]["conf"][:n_items]}
                for i in range(added_ids):
                    new_id = str(int(id_) + i + random.randint(10000,100000))
                    uniqued = [str(i) for i in unique_ids[i+1]]
                    previous_n = len(unique_ids[i])
                    n_items = len(uniqued)
                    corrected_ids[new_id] = {"frames":uniqued, "bbox_left":sequence_ids[id_]["bbox_left"][previous_n:previous_n+n_items], "bbox_top":sequence_ids[id_]["bbox_top"][previous_n:previous_n+n_items], "bbox_w":sequence_ids[id_]["bbox_w"][previous_n:previous_n+n_items], "bbox_h":sequence_ids[id_]["bbox_h"][previous_n:previous_n+n_items], "conf":sequence_ids[id_]["conf"][previous_n:previous_n+n_items]}
        except:
            print("Error in id: "+id_)
    return(corrected_ids)

def reordering_deepsort_corrected_conf(outfile,max_separation=2):
    deepsort_dict = reordering_deepsort_conf(outfile)
    sequence_ids=postprocessing_id_conf(dict1=deepsort_dict)
    corrected_id=correcting_mixed_ids_conf(sequence_ids=sequence_ids,max_separation=max_separation)

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


class Datasets:
    """
    The Datasets class aims for creating the dataset to be labelled for each of the Ash Dieback projects.
    """
    def __init__(self,input_dir,output_dir,project_name,n_images):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.project = project_name
        self.n_images = n_images
        self.yolo_model = r"\\gb010587mm\Software_dev\Ash_Dieback_Solution\ash_dieback_upgraded\weights\yolov5\best_1280.pt"
        self.deepsort_model = r"\\gb010587mm\Software_dev\Ash_Dieback_Solution\ash_dieback_solution\weights\deepsort\osnet_ibn_x1_0_imagenet.pth"
        self.config_deepsort = r"\\gb010587mm\Software_dev\Ash_Dieback_Solution\ash_dieback_solution\weights\deepsort\deep_sort_experimental.yaml"
        self.distribution_labels = {"20":0.1,"50":0.4,"70":0.3,"100":0.2}

    def listing_videos(self):
        # Creating an empty list
        vids = []
        # Populating the list with all the svi videos
        for path, subdirs, files in os.walk(self.input_dir):
            for name in files:
                if name.endswith(".svo"):
                    if name.split("_")[1]== "front":
                        vids.append(os.path.join(path, name))
        return(vids)

    def sampling_vids(self,videos,sample_size=0.20):
        """
        Provided a list with videos, it returns a random sample 
        """
        n_vids = len(videos)
        n_sample = int(math.ceil(sample_size*n_vids))
        sample = random.sample(videos,n_sample)

        return(sample,n_sample)

    def organising_outdir(self):
        # # checking whether there is a legacy directory there
        # if os.path.exists(os.path.join(self.output_dir,self.project)):
        #     shutil.rmtree(os.path.join(self.output_dir,self.project))  # delete output folder
        # creating a folder for the avi videos
        os.makedirs(os.path.join(self.output_dir,self.project,"avi_videos"),exist_ok=True)
        # creating a folder for the ash detection
        os.makedirs(os.path.join(self.output_dir,self.project,"ash_detection"),exist_ok=True)
        # defining number of slots
        n_subdir = int(math.ceil(self.n_images/3000))
        # defining name of slots
        slots = ["slot_"+str(i) for i in range(n_subdir)]
        # creating the slot folders
        for slot in slots:
            # we create the folder for the slot
            os.makedirs(os.path.join(self.output_dir,self.project,"ash_detection",slot),exist_ok=True)
            # we create a sub folder for the images
            os.makedirs(os.path.join(self.output_dir,self.project,"ash_detection",slot,slot+"_images"),exist_ok=True)
            # we create a sub folder for the labels
            os.makedirs(os.path.join(self.output_dir,self.project,"ash_detection",slot,"labels"),exist_ok=True)
        # creating a folder for health classification dataset
        os.makedirs(os.path.join(self.output_dir,self.project,"health_class"),exist_ok=True)
        # creating a folder for fp model
        os.makedirs(os.path.join(self.output_dir,self.project,"fp_assessment"),exist_ok=True)
        # creating a folder for the label studio videos
        os.makedirs(os.path.join(self.output_dir,self.project,"ls_videos"),exist_ok=True)

        return(os.path.join(self.output_dir,self.project,"avi_videos"))

    def turning_video(self,video_list):
        """
        This method transforms the list of videos provided from .svo to .avi format
        """
        # Iterating over all the videos selected
        for item in video_list:
            #transforming the video to avi
            fname = os.path.basename(item)
            try:
                avi_video(input_vid=item,output_video=os.path.join(self.output_dir,self.project,"avi_videos",fname[:-4]+".avi"),vid_resolution="2K")
            except:
                print("INVALID VIDEO - continue with next video")
                continue
    
    def turning_video_LS(self,video_list):
        """
        This method transforms the list of videos provided from .svo to .avi format
        """
        # Iterating over all the videos selected
        for item in video_list:
            #transforming the video to avi
            fname = os.path.basename(item)
            try:
                avi_video(input_vid=item,output_video=os.path.join(self.output_dir,self.project,"ls_videos","avi_videos",fname[:-4]+".avi"),vid_resolution="2K")
            except:
                print("INVALID VIDEO - continue with next video")
                continue
    
    def n_images_per_video(self, n_sample):
        """
        This method defines the number of images to be extracted from each video
        """
        n_images_video = int(self.n_images/n_sample)
        return(n_images_video)
    
    def select_folder(self,input_dir):
        files_count = dict()
        for root, dirs, files in os.walk(input_dir):
            files_count[root] = len(files)
        return(files_count)

    def extracting_images(self,vid_folder,n_sample):
        
        for vd in os.listdir(vid_folder):
            # inference and tracking on the video
            detect_and_track_conf(yolo_model=self.yolo_model,deep_sort_model=self.deepsort_model,config_deepsort=self.config_deepsort,imgsz=[640],out=self.output_dir,source=os.path.join(vid_folder,vd),conf_thres=0.2,iou_thres=0.45,classes=[0,2])
            try:
                # obtaining the file corrected
                deep_sort = reordering_deepsort_corrected_conf(outfile=os.path.join(self.output_dir,"Tracked.txt"))
            except:
                print("No detections in the entire video")
                continue
            # removing the txt from deepsort
            os.remove(os.path.join(self.output_dir,"Tracked.txt"))
            # create a dataframe from the deep sort dictionary
            df_ = pd.DataFrame.from_dict(deep_sort,orient='index')
            # randomise the dataframe
            df_random = df_.sample(frac=1,random_state=1)
            # convert the dataframe back to dictionary
            dict_sorted = df_random.to_dict(orient='index')
            # The number of images extracted per video
            n_img_to_extract = self.n_images_per_video(n_sample = n_sample)
            # assess whether the n sample is bigger than the frames with detections
            max_number = len(dict_sorted)
            if n_img_to_extract > max_number:
                n_img_to_extract = max_number

            # selecting keys to be saved
            n_50_max = int(self.distribution_labels["50"]*n_img_to_extract)
            n_70_max = int(self.distribution_labels["70"]*n_img_to_extract)
            n_100_max = int(self.distribution_labels["100"]*n_img_to_extract)

            n_50_count = 0
            n_70_count = 0
            n_100_count = 0
            keys_to_use = []

            # Obtaining detection info
            try:
                for it in dict_sorted:
                    if (float(dict_sorted[it]["conf"][0])) < 0.50:
                        if n_50_count == n_50_max:
                            continue
                        else:
                            n_50_count = n_50_count + 1
                            keys_to_use.append(it)
                    elif (float(dict_sorted[it]["conf"][0])) > 0.50 and (float(dict_sorted[it]["conf"][0]) < 0.70):
                        if n_70_count == n_70_max:
                            continue
                        else:
                            n_70_count = n_70_count + 1
                            keys_to_use.append(it)
                    if (float(dict_sorted[it]["conf"][0]) > 0.70) and (float(dict_sorted[it]["conf"][0]) < 1):
                        if n_100_count == n_100_max:
                            continue
                        else:
                            n_100_count = n_100_count + 1
                            keys_to_use.append(it)
            except:
                print("No detections in this video")
                continue

            # adding with no detection or less than 20% confidence
            n_20_max = int(self.distribution_labels["20"]*n_img_to_extract)

            range_list = [int(i) for i in dict_sorted]
            range_max = max(range_list)
            range_min = min(range_list)

            seq = list(range(range_min,range_max+1))
            random.shuffle(seq)
            n_20 = 0
            for i in seq:
                if n_20 == n_20_max:
                    break
                if i not in range_list:
                    keys_to_use.append(i)
                    n_20 = n_20 + 1
            
            # extract these images
            vidcap = cv2.VideoCapture(os.path.join(vid_folder,vd))
            success,image = vidcap.read()
            count = 0
            # loop over all the frames in the video
            keys_to_use = [int(ele) for ele in keys_to_use]
            while success:
                # evaluating whether the current frame is one of the frames to use
                if count in keys_to_use:
                    folder_to_use = None
                    # defining folder to save the image
                    fold = self.select_folder(input_dir=os.path.join(self.output_dir,self.project,"ash_detection"))
                    for kk in fold:
                        if kk.split("\\")[-1].endswith("_images"):
                            if fold[kk]<3000:
                                folder_to_use = kk
                                break
                        else:
                            continue
                    
                    if folder_to_use == None:
                        break
                    # evaluating brightness
                    color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    br_img = brightness(im_file=color_coverted)
                    # if too dark - brightnen and save it
                    if br_img < 40:
                        image_norm = cv2.normalize(image, None, alpha=0,beta=500, norm_type=cv2.NORM_MINMAX)
                        cv2.imwrite(os.path.join(folder_to_use,vd[:-4]+"_frame%d.png" % count), image_norm)
                    # if not - just save it 
                    else:
                        cv2.imwrite(os.path.join(folder_to_use,vd[:-4]+"_frame%d.png" % count), image) 
                    # evaluating whether it has predictions
                    if str(count) not in dict_sorted:
                        success,image = vidcap.read()
                        count += 1
                        continue
                    else:
                        # if the image contains prediction, the label is also saved
                        lbl_folder = os.path.join(Path(folder_to_use).parent.absolute(),"labels")
                        # saving the predicted label
                        with open(os.path.join(lbl_folder,vd[:-4]+"_frame%d.txt" % count), 'w') as f:
                            for idx,id_ in enumerate(dict_sorted[str(count)]["id"]):
                                img_width = image.shape[1]
                                img_height = image.shape[0]
                                clss = str(2)
                                x_centre = str((float(dict_sorted[str(count)]["bbox_left"][idx])/img_width) + (float(dict_sorted[str(count)]["bbox_w"][idx])/2)/img_width)
                                y_centre = str((float(dict_sorted[str(count)]["bbox_top"][idx])/img_height) + (float(dict_sorted[str(count)]["bbox_h"][idx])/2)/img_height)
                                w = str(float(dict_sorted[str(count)]["bbox_w"][idx])/img_width)
                                h = str(float(dict_sorted[str(count)]["bbox_h"][idx])/img_height)

                                f.write(clss+" "+x_centre+" "+y_centre+" "+w+" "+h+"\n")

                success,image = vidcap.read()
                print('Read a new frame: ', success)
                count += 1

    def obtaining_ls_videos(self, sample_videos=None,only_vid=False):
        if only_vid:
            # Obtaining a list of all videos 
            all_videos = self.listing_videos()
            # sampling videos
            videos,n_videos = self.sampling_vids(all_videos,sample_size=0.50)
            # turning these videos into avi
            os.makedirs(os.path.join(self.output_dir,self.project,"ls_videos","avi_videos"),exist_ok=True)
            self.turning_video_LS(videos)
            # preparing these videos
            for vv in os.listdir(os.path.join(self.output_dir,self.project,"ls_videos","avi_videos")):
                # defining the absolute path of the input video
                vd_abs_path = os.path.join(os.path.join(self.output_dir,self.project,"ls_videos","avi_videos",vv))
                # defining the absolute path of output video
                vd_abs_out_path = os.path.join(os.path.join(self.output_dir,self.project,"ls_videos","avi_videos",vv[:-4]+".mp4"))
                # adapting the image resolution
                cmd1 = "ffmpeg -i "+str(vd_abs_path)+" -vf scale=1280:720 -preset slow -crf 9 "+str(vd_abs_out_path)
                os.system(cmd1)
                # cropping the video
                out_final_vid = os.path.join(os.path.join(self.output_dir,self.project,"ls_videos",vv[:-4]+".mp4"))
                cmd2 = "ffmpeg -ss 1 -i "+str(vd_abs_out_path)+" -c copy -t 20 "+str(out_final_vid)
                os.system(cmd2)
            # removing the in-between videos
            shutil.rmtree(os.path.join(self.output_dir,self.project,"ls_videos","avi_videos"))
        else:
            # Obtaining a list of all videos 
            all_videos = self.listing_videos()
            # extracting those used for imagery
            all_rest_videos = list(set(all_videos) - set(sample_videos))
            # sampling videos
            videos,n_videos = self.sampling_vids(all_rest_videos,sample_size=0.10)
            # turning these videos into avi
            os.makedirs(os.path.join(self.output_dir,self.project,"ls_videos","avi_videos"),exist_ok=True)
            self.turning_video_LS(videos)
            # preparing these videos
            for vv in os.listdir(os.path.join(self.output_dir,self.project,"ls_videos","avi_videos")):
                # defining the absolute path of the input video
                vd_abs_path = os.path.join(os.path.join(self.output_dir,self.project,"ls_videos","avi_videos",vv))
                # defining the absolute path of output video
                vd_abs_out_path = os.path.join(os.path.join(self.output_dir,self.project,"ls_videos","avi_videos",vv[:-4]+".mp4"))
                # adapting the image resolution
                cmd1 = "ffmpeg -i "+str(vd_abs_path)+" -vf scale=1280:720 -preset slow -crf 9 "+str(vd_abs_out_path)
                os.system(cmd1)
                # cropping the video
                out_final_vid = os.path.join(os.path.join(self.output_dir,self.project,"ls_videos",vv[:-4]+".mp4"))
                cmd2 = "ffmpeg -ss 1 -i "+str(vd_abs_out_path)+" -c copy -t 30 "+str(out_final_vid)
                os.system(cmd2)
            # removing the in-between videos
            shutil.rmtree(os.path.join(self.output_dir,self.project,"ls_videos","avi_videos"))

class Video2Labels:
    """
    This class aims for creating YOLO labels out of video labelling.
    """
    def __init__(self,coco_labels,image_dir,output_dir):
        self.coco_labels = coco_labels
        self.image_dir = image_dir
        self.output_dir = output_dir


if __name__ == "__main__":
    """
    In order to define the labelling dataset for a project, we need to:
    
    1.- Define the input folder: a folder containing all the .svo videos from the project (e.g. all the valid .svo videos for Conwy)
    2.- Define the output folder: a folder where the output data (i.e. images to be labelled and predicted labels) will be output
    3.- Number of images to create (e.g. For Conwy project, how many images do we want to label before deploying the model there?)
    4.- sample_size - sample of videos that we consider representative (e.g. 15%). The number of images will be evenly obtained from this sample.
    """

    # input variables - that's the only bit of the code to be modified
    input_folder = r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\Surveyors_SD_Card_Data\Conwy\Conwy_part_2"
    output_folder = r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\Zed2i\training_sets\Validation\conwy"
    n_images = 500
    only_vid = False

    if only_vid: 
        #instantiating a dataset object
        dataset = Datasets(input_dir=input_folder,output_dir=output_folder,project_name="NYorkshire3",n_images=n_images)
        # extracting ls videos
        dataset.organising_outdir()
        dataset.obtaining_ls_videos(sample_videos=None,only_vid=True)

    else:
    
        #instantiating a dataset object
        dataset = Datasets(input_dir=input_folder,output_dir=output_folder,project_name="Conwy",n_images=n_images)

        # listing all the videos
        all_videos = dataset.listing_videos()

        # obtaining a random sample of videos
        sample_vids,n_sample = dataset.sampling_vids(videos=all_videos,sample_size=0.15)

        # organising output directory
        avi_folder = dataset.organising_outdir()

        # turning .svo videos into .avi
        dataset.turning_video(sample_vids)

        # extracting frames
        dataset.extracting_images(vid_folder=avi_folder,n_sample=n_sample)

        # extracting ls videos
        dataset.obtaining_ls_videos(sample_videos=sample_vids)