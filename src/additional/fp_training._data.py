import os
from pyexpat import model
from ssl import ALERT_DESCRIPTION_BAD_CERTIFICATE_HASH_VALUE
import cv2
import albumentations as A
import shutil
import json

# Adding extra functionality
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA) * (yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

class FpTraining():
    """
    This class generates an object that represents the training data for the FP model
    """
    def __init__(self,input_dir,gt_labels,hashed=False):
        self.input_dir = input_dir
        self.gt_labels = gt_labels
        self.hashed = hashed
    
    def listing_model_labels(self):
        model_labels = []
        for root, dirs, files in os.walk(os.path.join(self.input_dir,"ash_detection")):
            for it in files:
                if it.endswith(".txt"):
                    model_labels.append(os.path.join(root,it))
        return(model_labels)
    
    def listing_all_images(self):
        all_images = []
        for root, dirs, files in os.walk(os.path.join(self.input_dir,"ash_detection")):
            for it in files:
                if it.endswith(".jpg") or it.endswith(".png"):
                    all_images.append(os.path.join(root,it))
        return(all_images)
    
    def generating_healthClass(self, all_images):
        # Iterate over the ground truth labels
        for item in os.listdir(self.gt_labels):
            count = 0
            # read the label
            with open(os.path.join(self.gt_labels,item)) as fp:
                # line by line
                Lines = fp.readlines()
                for line in Lines:
                    count = count + 1
                    if len(line.split(" ")) < 2:
                        continue
                    # extract the information of the label
                    clss, centerx, centery, width, height = line.split(" ")
                    # find the corresponding image and get the shape
                    if self.hashed:
                        for mmg in all_images:
                            img_n = os.path.basename(mmg)
                            img_dir = os.path.dirname(mmg)
                            lbl_cc1 = item[9:-4]+".jpg"
                            lbl_cc2 = item[9:-4]+".png"
                            if (img_n == lbl_cc1) or (img_n == lbl_cc2):
                                img = cv2.imread(os.path.join(img_dir,mmg))
                                im_height, im_width, im_channels = img.shape
                                break
                            else:
                                continue
                    else:
                        for mmg in all_images:
                            img_n = os.path.basename(mmg)
                            img_dir = os.path.dirname(mmg)
                            lbl_cc1 = item[:-4]+".jpg"
                            lbl_cc2 = item[:-4]+".png"
                            if (img_n == lbl_cc1) or (img_n == lbl_cc2):
                                img = cv2.imread(os.path.join(img_dir,mmg))
                                im_height, im_width, im_channels = img.shape
                                break
                            else:
                                continue
                    
                    # extract the detection
                    x = int(float(centerx)*float(im_width))
                    y = int(float(centery)*float(im_height))
                    w = int(float(width)*float(im_width))
                    h = int(float(height)*float(im_height))

                    # adapting x and y
                    x = int(x - int(w/2))
                    y = int(y - int(y/2))

                    cropped_image = img[y:y+h,x:x+w]

                    # Apply albumentation
                    # transform = A.Compose([
                    #                 A.LongestMaxSize(max_size=224,interpolation=1),
                    #                 A.PadIfNeeded(min_height=224,min_width=224,border_mode=2)
                    #             ])
                    # transformed = transform(image=cropped_image)
                    # transformed_image = transformed["image"]

                    # saving image
                    if self.hashed:
                        img_name = item[9:-4]+"_"+str(count)+".png"
                    else:
                        img_name = item[:-4]+"_"+str(count)+".png"
                    cv2.imwrite(os.path.join(self.input_dir,"health_class",img_name),cropped_image)
    
    def unlabelled_images(self, all_images):
        # generating directory
        os.makedirs(os.path.join(self.input_dir,"No ash"),exist_ok=True)
        # generating a list with all the ground truth labels
        if self.hashed:
            lbl_list = os.listdir(self.gt_labels)
            lb_list = []
            for ll in lbl_list:
                lb_list.append(ll[9:])
        else:
            lb_list = os.listdir(self.gt_labels)
        # loop over all the images
        for item in all_images:
            img_nm = os.path.basename(item)
            img_dir = os.path.dirname(item)
            # get the name of the "presumed" corresponding label
            corr_lbl = img_nm[:-4]+".txt"
            # if there is no such label, we copy the image as having no ash tree
            if corr_lbl not in lb_list:
                shutil.copy(os.path.join(img_dir,img_nm),os.path.join(self.input_dir,"No ash",img_nm))
        
    def prepare_directory_fp(self):
        os.makedirs(os.path.join(self.input_dir,"fp_assessment","FP"),exist_ok=True)
        os.makedirs(os.path.join(self.input_dir,"fp_assessment","TP"),exist_ok=True)
    

    def FP_TP_training(self,list_labels,all_images):
        
        for item in all_images:
            # see whether there is a corresponding model label
            img_nm = os.path.basename(item)
            img_dir = os.path.dirname(item)
            lbl = img_nm[:-4]+".txt"

            for label in list_labels:
                lbl_name = os.path.basename(label)
                lbl_dir = os.path.dirname(label)

                # If the label corresponds with the image selected
                if lbl == lbl_name:
                    gt_label = None
                    for gt_lbl_ in os.listdir(self.gt_labels):
                        if self.hashed:
                            gt_lbl = gt_lbl_[9:]
                        else:
                            gt_lbl = gt_lbl_
                        if gt_lbl == lbl:
                            gt_label = gt_lbl_
                            break
                    if gt_label == None:
                        continue
                    with open(os.path.join(lbl_dir,lbl_name)) as fp:
                        # line by line
                        Lines = fp.readlines()
                        item_count = -1
                        for line in Lines:
                            item_count = item_count + 1
                            TP = []
                            if len(line.split(" ")) < 2:
                                continue
                            # extract the information of the label
                            clss, centerx, centery, width, height = line.split(" ")
                            img = cv2.imread(os.path.join(img_dir,img_nm))
                            im_height, im_width, im_channels = img.shape

                            BoxA = (int(float(centerx)*float(im_width))-int(float(width)*float(im_width)/2),int(float(centery)*float(im_height))-int(float(height)*float(im_height)/2),int(float(centerx)*float(im_width))+int(float(width)*float(im_width)/2),int(float(centery)*float(im_height))+int(float(height)*float(im_height)/2))
                            with open(os.path.join(self.gt_labels,gt_label)) as fp2:
                                # line by line
                                Lines2 = fp2.readlines()
                                for line2 in Lines2:
                                    if len(line2.split(" ")) < 2:
                                        continue

                                    # extract the information of the label
                                    clss2, centerx2, centery2, width2, height2 = line2.split(" ")
                                    BoxB = (int(float(centerx2)*float(im_width))-int(float(width2)*float(im_width)/2),int(float(centery2)*float(im_height))-int(float(height2)*float(im_height)/2),int(float(centerx2)*float(im_width))+int(float(width2)*float(im_width)/2),int(float(centery2)*float(im_height))+int(float(height2)*float(im_height)/2))

                                    # Calculate the IoU
                                    iou = bb_intersection_over_union(BoxA, BoxB)
                                    if iou > 0.5:
                                        # save as TP
                                        # extract the detection
                                        x = int(float(centerx)*float(im_width))
                                        y = int(float(centery)*float(im_height))
                                        w = int(float(width)*float(im_width))
                                        h = int(float(height)*float(im_height))
                                        # adapting x and y
                                        x = int(x - int(w/2))
                                        y = int(y - int(y/2))
                                        cropped_image = img[y:y+h,x:x+w]
                                        # Apply albumentation
                                        transform = A.Compose([
                                                        A.LongestMaxSize(max_size=224,interpolation=1),
                                                        A.PadIfNeeded(min_height=224,min_width=224,border_mode=2)
                                                    ])
                                        transformed = transform(image=cropped_image)
                                        transformed_image = transformed["image"]
                                        # saving image
                                        im_nm = img_nm[:-4]+"_"+str(item_count)+".jpg"
                                        cv2.imwrite(os.path.join(self.input_dir,"fp_assessment","TP",im_nm),transformed_image)
                                        # append to TP
                                        TP.append(1)
                                        break
                                        
                                if not TP:
                                    # save as TP
                                    # extract the detection
                                    x = int(float(centerx)*float(im_width))
                                    y = int(float(centery)*float(im_height))
                                    w = int(float(width)*float(im_width))
                                    h = int(float(height)*float(im_height))
                                    # adapting x and y
                                    x = int(x - int(w/2))
                                    y = int(y - int(y/2))
                                    cropped_image = img[y:y+h,x:x+w]
                                    # Apply albumentation
                                    transform = A.Compose([
                                        A.LongestMaxSize(max_size=224,interpolation=1),
                                        A.PadIfNeeded(min_height=224,min_width=224,border_mode=2)
                                                    ])
                                    transformed = transform(image=cropped_image)
                                    transformed_image = transformed["image"]
                                    # saving image
                                    im_nm = img_nm[:-4]+"_"+str(item_count)+".jpg"
                                    cv2.imwrite(os.path.join(self.input_dir,"fp_assessment","FP",im_nm),transformed_image)


def parse_json_health(input_images,json_label,out_folder):
    # opening the json file
    f = open(json_label)
    # dumping the file
    labelled_data = json.load(f)
    #processing health data
    for ix,item in enumerate(labelled_data):
        print("Dealing with item "+str(ix)+" out of "+str(len(labelled_data)))
        img_name = item["file_upload"][9:]
        try:
            lbl = item["annotations"][0]["result"][0]["value"]["choices"][0]
        except:
            continue

        if lbl ==  r"100% - 75% live crown":
            shutil.copy(os.path.join(input_images,img_name),os.path.join(out_folder,"0",img_name))
        elif lbl ==  r"75% - 50% live crown":
            shutil.copy(os.path.join(input_images,img_name),os.path.join(out_folder,"1",img_name))
        elif lbl ==  r"50% - 25% live crown":
            shutil.copy(os.path.join(input_images,img_name),os.path.join(out_folder,"2",img_name))
        elif lbl ==  r"25% - 0% live crown":
            shutil.copy(os.path.join(input_images,img_name),os.path.join(out_folder,"3",img_name))
        elif lbl ==  r"Dead":
            shutil.copy(os.path.join(input_images,img_name),os.path.join(out_folder,"3",img_name))
        elif lbl ==  r"No dieback":
            shutil.copy(os.path.join(input_images,img_name),os.path.join(out_folder,"0",img_name))
        else:
            continue


if __name__ == "__main__":


    FP_assessment = False

    if FP_assessment:

        input_dir=r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\Zed2i\training_sets\Ceredigion\Ceredigion" # directory with the 4 folders created for the data labelling of Monmouth
        gt_labels=r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\Zed2i\training_sets\Ceredigion\project-12-at-2022-11-11-09-27-5692a912\new_labels" #Labels downloaded from Label Studio
        hashed=False # whether the downloaded data has been hashed by Label Studio

        #Instantiate an object with Monmouth data
        monmouth_data = FpTraining(input_dir,gt_labels,hashed)
        # list of all model labels
        model_labels = monmouth_data.listing_model_labels()
        # listing all images
        all_images = monmouth_data.listing_all_images()
        # # generating health class images
        # monmouth_data.generating_healthClass(all_images=all_images)
        # # generating unlabelled images
        # monmouth_data.unlabelled_images(all_images=all_images)
        # Preparing directories for FP assessment
        monmouth_data.prepare_directory_fp()
        # Generating data for FP assessment
        monmouth_data.FP_TP_training(list_labels=model_labels,all_images=all_images)
    
    else:
        input_images = r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\Zed2i\training_sets\Glasgow\Glasgow\health_class"
        label_files = r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\Zed2i\training_sets\Glasgow\project-10-at-2022-11-16-12-03-27af4118.json" # download the json file
        out_folder = r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\Labelled_training_sets\health_class"

        parse_json_health(input_images=input_images,json_label=label_files,out_folder=out_folder)



                            






            


            


                    






    

    

