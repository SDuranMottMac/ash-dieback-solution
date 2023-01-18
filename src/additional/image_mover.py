
import argparse
import os
from osgeo import osr
from osgeo import ogr
import shutil
import numpy as np
import pandas as pd
import arcpy

class MissingPoints():
    '''This class takes pipeline outputs and finds all images (detections and no_bbox) that have not been used in the 
    shapefile output and move those images to their respective folders in a new SS folder in the pipeline outputs'''

    def __init__(self,input_path) -> None:
        self.input_path =input_path
        self.shp_folder = os.path.join(input_path,"Shapefile")
        self.detections_images = os.path.join(input_path,"Detections")

    def check_outputs(self):
        No_bbox = os.path.join(self.input_path,'No_bbox')
        if os.path.exists(No_bbox) and os.path.exists(self.shp_folder) and os.path.exists(self.detections_images):
            print('all folders good')
            return True

        return False

    def make_outputs(self):
        os.makedirs(os.path.join(self.input_path,'SS','Shapefile-original'))
        os.makedirs(os.path.join(self.input_path,"SS","non-used-detections"))
        os.makedirs(os.path.join(self.input_path,"SS","non-used-nbbox"))
        print('SS folder made')

    def img_list(self):
        print('finding list of images')
        img_list = []
        for item in os.walk(self.detections_images):
            for item in files:
                if item.endswith('.jpg'):
                    img_list.append(item)
        return img_list

    def table_to_data_frame(shapefile):
        #make attrubute table a df
        print('Making DF')
        arr = arcpy.da.TableToNumPyArray(shapefile)
        df = pd.DataFrame(arr)
        df = df.reset_index() 
        print(df) 
        return df

    def shp_list(self,df,img_list):
        #we only need a list of imgs to be moved. All images not in the shapefile need to be moved
        print('Finding images with points')
        for index,row in df:
            path = row['Path']
            img_name = path.split('\\')
            img_name = img_name[-1]
            if img_name in img_list:
                img_list.remove(img_name)
        return img_list
            
    def move_unused(self,unused_img):
        #set names for paths to be used
        non_detect = os.makedirs(os.path.join(self.input_path,"SS","non-used-detections"))
        non_nbbox = os.makedirs(os.path.join(self.input_path,"SS","non-used-nbbox"))
        nbbox = os.path.join(self.input_path,"No_bbox")

        #move images to folders
        print('Moving images')
        for img in unused_img:
            shutil.move(os.path.join(self.detections_images,img),os.path.join(non_detect,img))
            shutil.move(os.path.join(nbbox,img),os.path.join(non_nbbox,img))
                
    def img_mover(self):
        #check all directories are there 
        self.check_outputs()
        #make needed folders
        self.make_outputs()
        #obtain a list of names of images
        img_list = self.img_list()
        #duplicate shapefile before manipulation to SS folder
        og_shapefile = os.path.join(self.shp_folder,'merge.shp')
        arcpy.Copy_management(og_shapefile,os.path.join(self.input_path,'SS','Shapefile-original','merge-dup.shp'))
        #make attribute table into dataframe
        df = self.table_to_data_frame(og_shapefile)
        #find images that are not in shapefile
        unused_img = self.shp_list(df,img_list)
        #move unused images
        self.move_unused(unused_img)

if __name__ == "__main__":
    # Defining the parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline', type=str, default=None, help='input path for pipeline folder')
    opt = parser.parse_args()

    # Instantiating object
    image_finder = MissingPoints(input_path=opt.pipeline)
    # moving images
    image_finder.img_mover()

        


        
