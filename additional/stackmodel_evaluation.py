"""
This code aims to evaluate the performance of the stack of yolov5 model and the false positive model.
Therefore, obtaining an overall view of the performance of both together
"""
# importing libraries
import albumentations as A
import cv2
import imagesize
import json
import numpy as np
import os
from pathlib import Path
import platform
from podm import coco_decoder
from podm.metrics import get_pascal_voc_metrics, MetricPerClass, get_bounding_boxes
from podm.coco import PCOCOLicense, PCOCOInfo, PCOCOImage, PCOCOCategory, PCOCOBoundingBox, PCOCOSegments, \
    PCOCOObjectDetectionDataset
import sys
import shutil
import torch
import torch.backends.cudnn as cudnn

# importing Yolo dependencies
# ===========================
sys.path.insert(0, './models/Object_detection/yolov5')
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from models.common import DetectMultiBackend
from utils.torch_utils import select_device, smart_inference_mode
from utils.plots import Annotator, colors, save_one_box

# importing Timm dependencies
# ===========================
sys.path.insert(2, './models/Image_classification/pytorch-image-models')
from get_inference import inference_classification,get_fp_assessment,health_class_Resnet50

# adding extra functionality
# Create the annotations of the ECP dataset (Coco format)
coco_format = {"images": [{}], "categories": [], "annotations": [{}]}

def yolo_inference(image_directory,out_directory,weights,data_yml,conf_thres=0.25, iou_thres=0.45, classes=[0,2],view_img=False,save_conf=True,save_txt=True,save_images = False):

    # defining the source image directory
    source = str(image_directory)
    # defining whether we need to save the inference images
    save_img = save_images
    # checking whether the source directory is a file
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    # checking whether the source directory is an URL link
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    # checking whether the source directory is the webcam
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    # if the file is an URL or a .jpg/png, download it
    if is_url and is_file:
        source = check_file(source)  

    # Creating out directories
    save_dir = os.path.join(out_directory,"yolo_detections")  
    os.makedirs(save_dir,exist_ok=True)

    # Loading yolo model
    device=''
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data_yml, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = (1440, 1440)
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Yolo Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Running inference
    # model warmup
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    # looping through the dataset
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            # transforming to numpy and uploading to GPU
            im = torch.from_numpy(im).to(device)
            # turning pixel values into floating point
            im = im.half() if model.fp16 else im.float()  
            # normalising pixel values
            im /= 255  
            # expanding for batch dimension
            if len(im.shape) == 3:
                im = im[None]  

        # Inference on the image
        with dt[1]:
            visualize = False
            pred = model(im, augment=False, visualize=False)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic=False, max_det=1000)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  
            seen += 1
            if webcam:  
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            # path to p
            p = Path(p) 
            # defining image path
            save_path = os.path.join(save_dir,p.name)  
            # defining txt path
            txt_path = os.path.join(save_dir, p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  
            # defining image dimensions
            s += '%gx%g ' % im.shape[2:]
            # normalisation gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  
            imc = im0 
            # annotating images
            annotator = Annotator(im0, line_width=3, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # Write to file
                    if save_txt:  
                        # normalized xywh
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        # label format  
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    # Add bbox to image if saved
                    if save_img or False or view_img:
                        # integer class  
                        c = int(cls)  
                        # label
                        label = None if False else (names[c] if False else f'{names[c]} {conf:.2f}')
                        # use the annotator to add this info
                        annotator.box_label(xyxy, label, color=colors(c, True))

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
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
    return(save_dir)

def fp_assessment(image_directory, out_yolo, weights, out_directory,fp_conf = 0.8,file_ext = ".png"):
    # we need to get all outputs from yolo
    all_yolo_txt = os.listdir(out_yolo)
    # creating directory for fp assessment
    os.makedirs(os.path.join(out_directory,"fp_assessment"))
    # assess each item
    for item in all_yolo_txt:
        # fp file instantiated
        fp_file = []
        # find the corresponding image
        img_path = os.path.join(image_directory,item[:-4]+file_ext)
        # obtaining image characteristics
        img = cv2.imread(img_path)
        img_dimensions = img.shape
        img_height = img_dimensions[0]
        img_width = img_dimensions[1]
        if img is None:
            continue
        # read yolo detection file
        with open(os.path.join(out_yolo,item), 'r') as file_to_read:
            # loop over detections
            for line in file_to_read:
                # stripping line
                clss = line.split(" ")[0]
                x = line.split(" ")[1]
                y = line.split(" ")[2]
                w = line.split(" ")[3]
                h = line.split(" ")[4]
                conf = line.split(" ")[5]
                print("x: "+str(x)+" y: "+str(y)+" w: "+str(w)+" h: "+str(h)+" conf: "+str(conf))
                # denormalising data
                x = int(float(x)*float(img_width))
                y = int(float(y)*float(img_height))
                w = int(float(w)*float(img_width))
                h = int(float(h)*float(img_height))
                # obtaining cropped image
                cropped_image = img[int(y-int(h/2)):int(y+int(h/2)),int(x-int(w/2)):int(x+int(w/2))]
                # pre-processing the image for fp model
                transform = A.Compose([
                    A.LongestMaxSize(max_size=224,interpolation=1),
                    A.PadIfNeeded(min_height=224,min_width=224,border_mode=2)
                ])
                try:
                    transformed = transform(image=cropped_image)
                    transformed_image = transformed["image"] 
                    # assessing FP probability
                    fp_prob = get_fp_assessment(data_folder=transformed_image,model_path=weights)
                    fp_prob = np.average(np.array(fp_prob))
                    if fp_prob > fp_conf:
                        print("False positive")
                    else:
                        fp_file.append(line)
                except:
                    print("Error while reading image")
        # writing the detection file after fp assessment
        if not fp_file:
            print("No detections")
        else:
            with open(os.path.join(out_directory,"fp_assessment",item), 'w') as filehandle:
                for listitem in fp_file:
                    #filehandle.write('%s\n' % listitem)
                    filehandle.write(listitem)

    # return the fp assessment folder
    return(os.path.join(out_directory,"fp_assessment"))

def preparing_gt_data(gt_labels,outdir):
    # creating gt folder
    os.makedirs(os.path.join(outdir,"gt_data"),exist_ok=True)
    # copying data
    for item in os.listdir(gt_labels):
        shutil.copy(os.path.join(gt_labels,item),os.path.join(outdir,"gt_data",item))
    
    return(os.path.join(outdir,"gt_data"))

def verify_image_label(gt_directory,out_dir):
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []  # number (missing, found, empty, corrupt), message, segments
    os.makedirs(os.path.join(out_dir,"gt_checked_data"))

    for lb_f in os.listdir(gt_directory):
        try:
            lb_file = os.path.join(gt_directory,lb_f)
            # verify labels
            if os.path.isfile(lb_file):
                nf = 1  # label found
                with open(lb_file) as f:
                    lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                    lb = np.array(lb, dtype=np.float32)
                nl = len(lb)
                if nl:
                    assert lb.shape[1] == 5, f'labels require 5 columns, {lb.shape[1]} columns detected'
                    assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
                    assert (lb[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}'
                    _, i = np.unique(lb, axis=0, return_index=True)
                    if len(i) < nl:  # duplicate row check
                        lb = lb[i]  # remove duplicates
                    # creating the new label
                    with open(os.path.join(out_dir,"gt_checked_data",lb_f), 'w') as filehandle:
                        for listitem in lb:
                            #filehandle.write('%s\n' % listitem)
                            filehandle.write(listitem)
                else:
                    ne = 1  # label empty
                    lb = np.zeros((0, 5), dtype=np.float32)
            else:
                nm = 1  # label missing
                lb = np.zeros((0, 5), dtype=np.float32)

        except Exception as e:
            nc = 1
            msg = f'WARNING: ignoring corrupt image/label: {e}'

    # remove unchecked folder
    shutil.rmtree(gt_directory)
    return os.path.join(out_dir,"gt_checked_data")

def adding_images(img_folder, out_yolo, out_fp,out_gt,file_ext = ".png"):
    # adding images to yolo outcome
    for item in os.listdir(out_yolo):
        # finding image path
        img_path = os.path.join(img_folder,item[:-4]+file_ext)
        # copy the image
        shutil.copy(img_path,os.path.join(out_yolo,item[:-4]+file_ext))
    # adding to fp outcome
    for item in os.listdir(out_fp):
        # finding image path
        img_path = os.path.join(img_folder,item[:-4]+file_ext)
        # copy the image
        shutil.copy(img_path,os.path.join(out_fp,item[:-4]+file_ext))
    # adding for the gt data
    for item in os.listdir(out_gt):
    # finding image path
        img_path = os.path.join(img_folder,item[:-4]+file_ext)
        # copy the image
        shutil.copy(img_path,os.path.join(out_fp,item[:-4]+file_ext))

def create_image_annotation(file_path: Path, width: int, height: int, image_id: int):

    file_path = file_path.name
    image_annotation = {
        "file_name": file_path,
        "height": height,
        "width": width,
        "id": image_id,
    }
    return image_annotation


def create_annotation_from_yolo_format(
    min_x, min_y, width, height, image_id, category_id, annotation_id, segmentation=True
):
    bbox = (float(min_x), float(min_y), float(width), float(height))
    area = width * height
    max_x = min_x + width
    max_y = min_y + height
    if segmentation:
        seg = [[min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]]
    else:
        seg = []
    annotation = {
        "id": annotation_id,
        "image_id": image_id,
        "bbox": bbox,
        "area": area,
        "iscrowd": 0,
        "category_id": category_id,
        "segmentation": seg,
    }

    return annotation

def get_images_info_and_annotations(input_directory):
    path = Path(input_directory)
    annotations = []
    images_annotations = []
    if path.is_dir():
        file_paths = sorted(path.rglob("*.jpg"))
        file_paths += sorted(path.rglob("*.jpeg"))
        file_paths += sorted(path.rglob("*.png"))
    else:
        with open(path, "r") as fp:
            read_lines = fp.readlines()
        file_paths = [Path(line.replace("\n", "")) for line in read_lines]

    image_id = 0
    annotation_id = 1  # In COCO dataset format, you must start annotation id with '1'

    for file_path in file_paths:
        # Check how many items have progressed
        print("\rProcessing " + str(image_id) + " ...", end='')

        # Build image annotation, known the image's width and height
        w, h = imagesize.get(str(file_path))
        image_annotation = create_image_annotation(
            file_path=file_path, width=w, height=h, image_id=image_id
        )
        images_annotations.append(image_annotation)

        label_file_name = f"{file_path.stem}.txt"
        if False:
            annotations_path = file_path.parent / YOLO_DARKNET_SUB_DIR / label_file_name
        else:
            annotations_path = file_path.parent / label_file_name

        if not annotations_path.exists():
            continue  # The image may not have any applicable annotation txt file.

        with open(str(annotations_path), "r") as label_file:
            label_read_line = label_file.readlines()

        # yolo format - (class_id, x_center, y_center, width, height)
        # coco format - (annotation_id, x_upper_left, y_upper_left, width, height)
        for line1 in label_read_line:
            label_line = line1
            category_id = (
                int(label_line.split()[0]) + 1
            )  # you start with annotation id with '1'
            x_center = float(label_line.split()[1])
            y_center = float(label_line.split()[2])
            width = float(label_line.split()[3])
            height = float(label_line.split()[4])

            float_x_center = w * x_center
            float_y_center = h * y_center
            float_width = w * width
            float_height = h * height

            min_x = int(float_x_center - float_width / 2)
            min_y = int(float_y_center - float_height / 2)
            width = int(float_width)
            height = int(float_height)

            annotation = create_annotation_from_yolo_format(
                min_x,
                min_y,
                width,
                height,
                image_id,
                category_id,
                annotation_id,
                segmentation=False,
            )
            annotations.append(annotation)
            annotation_id += 1

        image_id += 1  # if you finished annotation work, updates the image id.

    return images_annotations, annotations

def yolo2coco(input_dir,out_file,name="yolo_out.json"):

    # output variables
    output_name = name
    output_path = os.path.join(out_file,output_name)

    # classes
    classes = [
    "Other Dead Trees",
    "Distant Ash Tree",
    "Ash tree",
    "Immature Ash Tree"
    ]

    (
        coco_format["images"],
        coco_format["annotations"],
    ) = get_images_info_and_annotations(input_dir)

    for index, label in enumerate(classes):
        categories = {
            "supercategory": "Tree",
            "id": index + 1,  # ID starts with '1' .
            "name": label,
        }
        coco_format["categories"].append(categories)

    coco_format["info"] = {
        "contributor":"Sergio Duran", 
        "description":"Ash Dieback evaluation",
        "version":"1.0",
        "year":2022,
        "url":"www.mottmac.com",
        "date_created":" "}
    
    coco_format["licenses"] = [
        {
            "url":"www.mottmac.com",
            "id":1,
            "name": "Mottmac License"

        }
    
    ]
    
    with open(output_path, "w") as outfile:
        json.dump(coco_format, outfile, indent=4)

    print("Finished!")
    return(os.path.join(out_file,output_name))

def calculating_metrics(coco_json,gt_json):
    with open(gt_json) as fp:
        gt_dataset = coco_decoder.load_true_object_detection_dataset(fp)
    with open(coco_json) as fp:
        pred_dataset = coco_decoder.load_pred_object_detection_dataset(fp, gt_dataset)

    gt_BoundingBoxes = get_bounding_boxes(gt_dataset)
    pd_BoundingBoxes = get_bounding_boxes(pred_dataset)

    results = get_pascal_voc_metrics(gt_BoundingBoxes, pd_BoundingBoxes, .5)

    mAP = MetricPerClass.mAP(results)

    return(mAP)

if __name__ == "__main__":

    # variables to define
    img_directory = r"\\gb010587mm\Software_dev\Ash_Dieback_Solution\Tests\evaluation\images"
    out_directory = r"\\gb010587mm\Software_dev\Ash_Dieback_Solution\Tests\test1"
    yolo_weights = r"\\gb010587mm\Software_dev\Ash_Dieback_Solution\ash_dieback_upgraded\weights\yolov5\best_1280.pt"
    fp_weights = r"\\gb010587mm\Software_dev\Ash_Dieback_Solution\ash_dieback_upgraded\weights\Resnet_50\FP_assessment.keras"
    data_yml = r"\\gb010587mm\Software_dev\Ash_Dieback_Solution\ash_dieback_solution\models\Object_detection\yolov5\data\Ash_dieback.yaml"
    gt_labels = r"\\gb010587mm\Software_dev\Ash_Dieback_Solution\Tests\evaluation\labels"

    # Obtaining yolo detections
    yolo_assessment = yolo_inference(image_directory=img_directory,out_directory=out_directory,weights=yolo_weights,data_yml=data_yml,conf_thres=0.25, iou_thres=0.45, classes=[0,2],view_img=False,save_conf=True,save_txt=True,save_images = False)
    # Assessing FP in these detections
    fp_assessment_folder = fp_assessment(image_directory=img_directory, out_yolo=yolo_assessment, weights=fp_weights, out_directory=out_directory,fp_conf = 0.9,file_ext = ".png")
    # preparing output sets
    gt_directory = preparing_gt_data(gt_labels=gt_labels,outdir=out_directory)
    checked_gt = verify_image_label(gt_directory,out_directory)
    adding_images(img_folder=img_directory, out_yolo=yolo_assessment, out_fp=fp_assessment_folder,out_gt=checked_gt,file_ext = ".png")
    # calculating the yolo_out coco file, as well as the fp assessment coco file
    yolo_2_coco_path = yolo2coco(input_dir=yolo_assessment,out_file=out_directory,name="yolo_out.json")
    fp_2_coco_path = yolo2coco(input_dir=fp_assessment_folder,out_file=out_directory,name="fp_out.json")
    gt_2_coco_path = yolo2coco(input_dir=checked_gt,out_file=out_directory,name="gt_out.json")
    # calculating metrics
    mAP_yolo = calculating_metrics(yolo_2_coco_path,gt_2_coco_path)
    mAP_fpAssessment = calculating_metrics(fp_2_coco_path,gt_2_coco_path)

 
    print("Yolo mAP "+str(mAP_yolo))
    print("After FP assessment mAP "+str(mAP_fpAssessment))
    print("which represents an improvement of "+str(mAP_fpAssessment-mAP_yolo)+" mAP")