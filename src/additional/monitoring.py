'''
This code is used to monitor the performance of the system in production.
The system is generating a "monitoring" folder per batch. This folder contains images and labels.
1) Upload the images to label studio and label them.
2) Export the labels as Yolo format.
3) Provide these labels and the labels output by the system as input.
4) The system will generate a report.
'''
import coco_tools
import os
import numpy as np
import cv2
import shutil
import json

def adapting_name(reviewed_labels_path,images_path):
    # obtaining a list of reviewed files and images
    reviewed_files = os.listdir(reviewed_labels_path)
    images = os.listdir(images_path)

    for item in reviewed_files:
        # setting new name as name without hash key
        new_name = os.path.basename(item)[9:]
        # renaming the file
        os.rename(os.path.join(reviewed_labels_path,item),os.path.join(reviewed_labels_path,new_name))

    for item in images:
        # setting new name as name without hash key
        new_name = os.path.basename(item)[9:]
        # renaming the file
        os.rename(os.path.join(images_path,item),os.path.join(images_path,new_name))

def adapting_values(reviewed_labels_path,model_labels_path,images_path):
    # listing all reviewed files
    reviewed_files = os.listdir(reviewed_labels_path)
    # creating a list to be populated with files to be removed
    to_remove = []
    # iterating over the files
    for file_ in reviewed_files:
        # add file to the list of objects-to-be-removed
        to_remove.append(os.path.join(reviewed_labels_path,file_[:-4]+"_copy.txt"))
        # moving and renaming the file
        shutil.move(os.path.join(reviewed_labels_path,file_),os.path.join(reviewed_labels_path,file_[:-4]+"_copy.txt"))
        # opening a new file to be populated
        destination = open(os.path.join(reviewed_labels_path,file_), "w")
        # opening the old file
        source = open(os.path.join(reviewed_labels_path,file_[:-4]+"_copy.txt"), "r")
        # obtaining info about the image
        image_name = os.path.join(images_path,file_[:-4]+".png")
        img = cv2.imread(image_name)
        height, width, channels = img.shape
        # reading the source file
        c = source.read()
        if c == '':
            continue
        content = c.split('\n')
        # removing the file bit of the content (which is an empty space)
        content = content[:-1]
        # writing the destination file
        for line in content:
            split = line.split(' ')
            cl = split[0]
            bbox = [int(round(float(split[1])*width)-(round(float(split[3])*width)/2)),int(round(float(split[2])*height)-(round(float(split[4])*height)/2)),int(round(float(split[1])*width)+(round(float(split[3])*width)/2)),int(round(float(split[2])*height)+(round(float(split[4])*height)/2))]
            new_line = cl+" "+str(bbox[0])+" "+str(bbox[1])+" "+str(bbox[2])+" "+str(bbox[3])+"\n"
            destination.write(new_line)
        
        source.close()
        destination.close()
    # removing original files (to be removed)  
    for items in to_remove:
        os.remove(items)

    # same process with labels coming from the yolo model
    model_files = os.listdir(model_labels_path)
    to_remove2 = []
    for file_ in model_files:
        to_remove2.append(os.path.join(model_labels_path,file_[:-4]+"_copy.txt"))
        shutil.move(os.path.join(model_labels_path,file_),os.path.join(model_labels_path,file_[:-4]+"_copy.txt"))
        destination = open(os.path.join(model_labels_path,file_), "w")
        source = open(os.path.join(model_labels_path,file_[:-4]+"_copy.txt"), "r")
        image_name = os.path.join(images_path,file_[:-4]+".png")
        img = cv2.imread(image_name)
        height, width, channels = img.shape
        c = source.read()
        if c == '':
            continue
        content = c.split('\n')
        content = content[:-1]
        #content.remove('')
        for line in content:
            split = line.split(' ')
            cl = split[0]
            score = float(split[1])/100
            bbox = [int(round(float(split[2])*width)-(round(float(split[4])*width)/2)),int(round(float(abs(split[3]))*height)-(round(float(split[5])*height)/2)),int(round(float(split[2])*width)+(round(float(split[4])*width)/2)),int(round(float(abs(split[3]))*height)+(round(float(split[5])*height)/2))]
            new_line = cl+" "+str(score)+" "+str(bbox[0])+" "+str(bbox[1])+" "+str(bbox[2])+" "+str(bbox[3])+"\n"
            destination.write(new_line)
        
        source.close()
        destination.close()

        for items in to_remove2:
            os.remove(items)


def obtaining_mAP(reviewed_labels_path,model_labels_path,categories):
    det_files = [os.path.join(model_labels_path, f) for f in os.listdir(model_labels_path) if f.endswith('.txt')]
    gt_files = [os.path.join(reviewed_labels_path, f) for f in os.listdir(reviewed_labels_path) if f.endswith('.txt')]

    image_ids_det = []
    image_ids_gt = []
    gt_boxes = []
    gt_classes = []
    det_boxes = []
    det_classes = []
    det_scores = []

    #Read ground truth files to list
    for file_name in gt_files:
        file = open(file_name, 'r')
        c = file.read()
        if c == '':
            continue

        content = c.split('\n')
        #content.remove('')
        content = content[:-1]
        gt_b = []
        gt_c = []
        for line in content:
            split = line.split(' ')
            cl = split[0]
            bbox = np.array([int(x) for x in split[1:]])
            gt_c.append(0) 
            gt_b.append(bbox)
        im_id = file_name.split('\\')[-1].replace('.txt', '')
        image_ids_gt.append(im_id) 
        gt_boxes.append(np.array(gt_b))
        gt_classes.append(np.array(gt_c))
        file.close()

    #Read detections files to list
    for file_name in det_files:
        file = open(file_name, 'r')
        c = file.read()
        if c == '':
            continue

        content = c.split('\n')
        #content.remove('')
        content = content[:-1]
        det_b = []
        det_c = []
        det_s = []
        for line in content:
            split = line.split(' ')
            cl = split[0]
            score = float(split[1])
            bbox = np.array([int(x) for x in split[2:]])
            det_c.append(0) 
            det_b.append(bbox)
            det_s.append(score)
        im_id = file_name.split('\\')[-1].replace('.txt', '')
        image_ids_det.append(im_id) 
        det_boxes.append(np.array(det_b))
        det_scores.append(np.array(det_s))
        det_classes.append(np.array(det_c))
        file.close()

    #Convert all lists to numpy arrays
    image_ids_gt = np.array(image_ids_gt)
    gt_boxes = np.array(gt_boxes)
    gt_classes = np.array(gt_classes)
    image_ids_det = np.array(image_ids_det)
    det_boxes = np.array(det_boxes)
    det_classes = np.array(det_classes)

    #Convert ground truth list to dict
    groundtruth_dict = coco_tools.ExportGroundtruthToCOCO(
        image_ids_gt, gt_boxes, gt_classes,
        categories)
    print(groundtruth_dict)
    #Convert detections list to dict
    detections_list = coco_tools.ExportDetectionsToCOCO(
        image_ids_det, det_boxes, det_scores,
        det_classes, categories)
    print(detections_list)
    groundtruth = coco_tools.COCOWrapper(groundtruth_dict)
    detections = groundtruth.LoadAnnotations(detections_list)
    evaluator = coco_tools.COCOEvalWrapper(groundtruth, detections,
                                        agnostic_mode=False)
    metrics, empty = evaluator.ComputeMetrics()
    return(metrics)

def saving_metrics(metrics,out_folder):
    json_object = json.dumps(metrics,indent=3)
    os.path.dirname(out_folder)
    with open(os.path.join(os.path.dirname(out_folder),"metrics.json"),'w') as outfile:
        outfile.write(json_object)
if __name__ == '__main__':
    # The path of the labels reviewed by the expert - you have downloaded them from label studio in YOLO format
    reviewed_labels_path = r'C:\Users\DUR94074\OneDrive - Mott MacDonald\Desktop\example22\project-3-at-2022-04-06-10-09-023adcfe\labels'
    # The path of the labels output by the system - they have been saved under the "Monitoring" folder
    model_labels_path = r'C:\Users\DUR94074\OneDrive - Mott MacDonald\Desktop\example22\project-3-at-2022-04-06-10-09-023adcfe\model_labels'
    # The path to the images - these are the images downloaded from Label Studio (important to put these images)
    images_path = r'C:\Users\DUR94074\OneDrive - Mott MacDonald\Desktop\example22\project-3-at-2022-04-06-10-09-023adcfe\images'
    # The categories of the dataset.
    categories = np.array([{'id': 0, 'name': 'AshTree'}])

    adapting_name(reviewed_labels_path=reviewed_labels_path,images_path=images_path)
    adapting_values(reviewed_labels_path=reviewed_labels_path,model_labels_path=model_labels_path,images_path=images_path)
    metrics = obtaining_mAP(reviewed_labels_path=reviewed_labels_path,model_labels_path=model_labels_path,categories=categories)
    saving_metrics(metrics=metrics,out_folder=model_labels_path)