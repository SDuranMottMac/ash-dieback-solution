'''This is a post processing version of deploying a FP model. Takes a pipeline output (must include: Detections, No_bbox directories.
It will make a cropped detections folder and delete it when done. Will not be deleted if --stat argument is used
when --stat argument is used, the output will be the proportion of outputs that would be moved. (best for testing) 
and a scatter plot of preditions'''
import os
import tensorflow as tf
from tensorflow import keras
import torch
import shutil
import cv2 
import numpy as np
import albumentations as A
import argparse
import matplotlib.pyplot as plt

class FP_assessment():
    def __init__(self,pipeline_outputs,model):
        self.pipeline_outputs = pipeline_outputs
        self.model = model
    
    def crop_img(self):
        '''This finds the bbox in a detections image (set to [0,255,255]) then crops the respective image in No_bbox. Modifies 
        cropped image for the prediction. Saves image in /pipeline_outputs/Cropped'''
        detections_p = os.path.join(self.pipeline_outputs,'Detections')
        nobbox_p = os.path.join(self.pipeline_outputs,'No_bbox')
        if os.path.isdir(detections_p) and os.path.isdir(nobbox_p): #both Detections and no_bbox must exist
            #make Cropped directory for storing cropped images
            path = os.path.join(self.pipeline_outputs,'Cropped') 
            os.makedirs(path)
            print(' made ',path)

            color = np.array([0,255,255]) #BGR - colour of bbox to find
            for root, folders, files in os.walk(detections_p,topdown=True):
                for item in files:
                    if item.endswith(".jpg") or item.endswith(".png"):
                        #crop image 
                        box_img = cv2.imread(os.path.join(root,item))
                        mask = cv2.inRange(box_img, color,color)
                        contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        try:
                            img = cv2.imread(os.path.join(nobbox_p,item))
                        except:
                            print('missing img, skipping')
                            continue
                        for cnt in contours:
                            x,y,w,h = cv2.boundingRect(cnt)
                            cropped1 = img[y:y+h,x:x+w]

                        #make cropped image ready for assessment
                        try:
                            transform = A.Compose([
                                        A.LongestMaxSize(max_size=224,interpolation=1),
                                        A.PadIfNeeded(min_height=224,min_width=224,border_mode=2)
                                        ])
                            transformed = transform(image=cropped1)
                            transformed_image = transformed["image"]
                            cv2.imwrite(os.path.join(path,item),transformed_image)

                        except:
                            #if there is an error, will exit with assessment of what could have caused it
                            print('contours array',contours)
                            cv2.imwrite(os.path.join(self.pipeline_outputs,item),mask)
                            print('most likely missing contour, mask image made,check color of bbox (line32)')
                            exit()           
        else:
            print('missing folder')

    def move_img(self,thresh = .5):
        ''' Move potential FP detection to folder based on threshold'''
        FPmodel = keras.models.load_model(self.model)
        image_size = (224, 224)
        cropped_folder = os.path.join(self.pipeline_outputs,'Cropped')
        detections_p = os.path.join(self.pipeline_outputs,'Detections')

        if os.path.isdir(cropped_folder):
            path = os.path.join(self.pipeline_outputs,'Potential False Positives')
            if os.path.exists(path):
                #move all file contents of Potential False Positives back to Detections
                for item in os.listdir(path):
                    png_path = os.path.join(detections_p,item)
                    shutil.move(os.path.join(path,item),png_path)
                #remove to be sure there is nothing random
                shutil.rmtree(path)
            #make Potential False Positives directory to store predictions  
            os.makedirs(path)
            print('made ', path)

            for root, folders, files in os.walk(cropped_folder,topdown=True):
                for item in files:
                    if item.endswith(".jpg") or item.endswith(".png"):
                        test_image = keras.utils.load_img(os.path.join(root,item), target_size=image_size)
                        test_image = keras.utils.img_to_array(test_image)
                        test_image = np.expand_dims(test_image,axis=0)
                        results = FPmodel.predict(test_image)
                        if results.item(0) >= thresh:
                            #move imagae to Potential False Positives directory if it is higher than threshold 
                            #higher result is more FP, lower is TP
                            print('moving image')
                            png_path = os.path.join(detections_p,item)
                            shutil.move(png_path,os.path.join(path,item))
    
    def move_img_lowest(self):
        '''Move the 20% most likely to be FP to new folder'''
        #load model and set image size to what model expects
        FPmodel = keras.models.load_model(self.model)
        image_size = (224, 224)
        cropped_folder = os.path.join(self.pipeline_outputs,'Cropped')
        detections_p = os.path.join(self.pipeline_outputs,'Detections')
        prob_dict = {}
        if os.path.isdir(cropped_folder):
            path = os.path.join(self.pipeline_outputs,'Potential False Positives')
            if os.path.exists(path):
                #move all file contents of Potential False Positives back to Detections
                for item in os.listdir(path):
                    png_path = os.path.join(detections_p,item)
                    shutil.move(os.path.join(path,item),png_path)
                #remove to be sure there is nothing random
                shutil.rmtree(path)
            #make Potential False Positives directory to store predictions  
            os.makedirs(path)
            print('made ', path)

            for root, folders, files in os.walk(cropped_folder,topdown=True):
                for item in files:
                    if item.endswith(".jpg") or item.endswith(".png"):
                        test_image = keras.utils.load_img(os.path.join(root,item), target_size=image_size)
                        test_image = keras.utils.img_to_array(test_image)
                        test_image = np.expand_dims(test_image,axis=0)
                        results = FPmodel.predict(test_image)
                        prob_dict[item] = results.item(0) #store results in dictionary: image name:results
            print('collected all results')
            
            #sort dictionary in most likely to be FP to least likely
            prob_dict_sort = sorted(prob_dict,key = prob_dict.get,reverse=True) 
            #What is the number of images we are wanting 
            lowest_prop = int(len(prob_dict)*.2)
            for key in prob_dict_sort[:lowest_prop+1]:
                #move the number of images we want
                detect = os.path.join(detections_p,key)
                print('moving image')
                shutil.move(detect,os.path.join(path,key))
    
    def FP_stat(self,thresh):
        '''Returns statistics on what would be moved in move_img'''
        FP_prob = 0
        FP_res = []
        #load keras model and set image_size to what the model expects
        FPmodel = keras.models.load_model(self.model)
        image_size = (224, 224) 
        cropped_folder = os.path.join(self.pipeline_outputs,'Cropped')
        detections_p = os.path.join(self.pipeline_outputs,'Detections')
        if os.path.isdir(cropped_folder):
            for root, folders, files in os.walk(cropped_folder,topdown=True):
                for item in files:
                    if item.endswith(".jpg") or item.endswith(".png"):
                        test_image = keras.utils.load_img(os.path.join(root,item), target_size=image_size)
                        test_image = keras.utils.img_to_array(test_image)
                        test_image = np.expand_dims(test_image,axis=0)
                        results = FPmodel.predict(test_image)
                        FP_res.append(results.item(0))
                        if results.item(0) >= thresh:
                            FP_prob += 1

            num_detect = len(os.listdir(detections_p))
            return [num_detect,FP_prob,float(FP_prob/num_detect),FP_res]
    
    def conf_compare(self):
        FP_conf = []
        pipe_conf = []
        FPmodel = keras.models.load_model(self.model)
        image_size = (224, 224) 
        cropped_folder = os.path.join(self.pipeline_outputs,'Cropped')
        detections_p = os.path.join(self.pipeline_outputs,'Detections')
        if os.path.isdir(cropped_folder):
            for root, folders, files in os.walk(cropped_folder,topdown=True):
                for item in files:
                    if item.endswith(".jpg") or item.endswith(".png"):
                        #FP confidence
                        test_image = keras.utils.load_img(os.path.join(root,item), target_size=image_size)
                        test_image = keras.utils.img_to_array(test_image)
                        test_image = np.expand_dims(test_image,axis=0)
                        results = FPmodel.predict(test_image)
                        FP_conf.append(results.item(0))

                        #pipeline confidence
                        png_name = item.split('_')
                        conf = png_name[-1]
                        conf = conf.replace('.png','')
                        pipe_conf.append(float(conf))

        plt.scatter(pipe_conf,FP_conf)
        plt.show()

    def run_fp(self,stat):
        if not os.path.exists(os.path.join(self.pipeline_outputs,'Cropped')):
            #will only run if there are not already cropped images  
            print('Cropping images')
            self.crop_img()
        
        if stat == True:
            print('Entering stat')
            results = self.FP_stat()
            print('Number of detections, number that would be moved, proportion moved',results[:-1])
            #plot the results
            lis = results[3]
            x = [lis.index(i) for i in lis]
            plt.scatter(x,lis)
            plt.show()
            #will not remove cropped folder under the assumption more stats will be ran - save processing time
            print('Did not remove cropped folder')
        
        else:
            print('Moving FP')
            self.move_img_lowest() #change depending on results wanted
            print('Removing Cropped folder')
            #remove cropped folder under the assumption outputs will be sufficient
            shutil.rmtree(os.path.join(self.pipeline_outputs,'Cropped'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline_out', type=str,help='Pipeline output folder')
    parser.add_argument('--FP_model',type=str,help='Keras FP model')
    parser.add_argument('--stat',action="store_true")
    opt = parser.parse_args()

    FP = FP_assessment(pipeline_outputs=opt.pipeline_out,model=opt.FP_model)
    FP.run_fp(opt.stat)

    '''Example Cmd line:
        python 'path to this script' --pipeline_out 'path to pipeline output to assess' --FP_model 'path to .keras model --stat (if running stat)' 
    '''
