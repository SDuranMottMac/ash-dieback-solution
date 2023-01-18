import argparse
import os
from osgeo import osr
from osgeo import ogr
from osgeo import gdal
from glob import glob
import shutil

def check_outputs(input_path):
        No_bbox = os.path.join(input_path,'No_bbox')
        if os.path.exists(No_bbox) and os.path.exists(os.path.join(input_path,"Shapefile")) and os.path.exists(os.path.join(input_path,"Detections")):
            print('all folders good')
            return True

        return False

class MissingPoints():
    '''This class takes pipeline outputs and finds all images (detections and no_bbox) that have not been used in the 
    shapefile output and move those images to their respective folders in a new SS folder in the pipeline outputs'''

    def __init__(self,input_path,conf) -> None:
        self.input_path =input_path
        self.shp_folder = os.path.join(input_path,"Shapefile")
        self.detections_images = os.path.join(input_path,"Detections")
        self.conf = conf

    def make_outputs(self):
        if not os.path.exists(os.path.join(self.input_path,"SS","non-used-detections")):
            #os.makedirs(os.path.join(self.input_path,'SS','Shapefile-original'))
            os.makedirs(os.path.join(self.input_path,"SS","non-used-detections"))
            os.makedirs(os.path.join(self.input_path,"SS","non-used-nbbox"))
        print('SS folder made')

    def img_list(self):
        print('finding list of images')
        img_list = []
        for root, dirs, files in os.walk(self.detections_images):
            for item in files:
                if item.endswith('.jpg') or item.endswith('.png'):
                    img_list.append(item)
        return img_list

    def shp_reader(self, shapefile, img_list):
        driver =ogr.GetDriverByName('ESRI Shapefile')
        table = driver.Open(shapefile)
        layer = table.GetLayer(0)
        #find paths in feature that coorespond with images
        for feature in layer:
            path = feature.GetField('Path')
            img_name = path.split('\\')
            img_name = img_name[-1]
            if img_name in img_list:
                #remove an image in the list if it is in shapefile
                img_list.remove(img_name)
        layer.ResetReading()
        #return list of images we need to move
        return img_list

    def shp_reader_conf(self,shapefile, img_list):
        review = []
        driver =ogr.GetDriverByName('ESRI Shapefile')
        table = driver.Open(shapefile)
        layer = table.GetLayer(0)
        #find paths in feature that coorespond with images
        for feature in layer:
            path = feature.GetField('Path')
            confidence = feature.GetField('Confidence')
            img_name = path.split('\\')
            img_name = img_name[-1]
            if img_name in img_list:
                #check if it needs to be reviewed 
                if confidence > self.conf:
                    review.append(img_name)
                #remove an image in the list if it is in shapefile,
                img_list.remove(img_name)
                continue
           
            exit()
        layer.ResetReading()
        #return list of images we need to move
        return img_list,review

    def move_unused(self,unused_img):
        #set names for paths to be used
        non_detect = os.path.join(self.input_path,"SS","non-used-detections")
        non_nbbox = os.path.join(self.input_path,"SS","non-used-nbbox")
        nbbox = os.path.join(self.input_path,"No_bbox")

        #move images to folders
        print('Moving unused images')
        for img in unused_img:
            try:
                shutil.move(os.path.join(self.detections_images,img),os.path.join(non_detect,img))
                shutil.move(os.path.join(nbbox,img),os.path.join(non_nbbox,img))
            except:
                shutil.move(os.path.join(self.detections_images,'False Positive',img),os.path.join(non_detect,img))
                shutil.move(os.path.join(nbbox,img),os.path.join(non_nbbox,img))
    
    def move_to_review(self,review):
        #only used if confidence is used
        detec = os.path.join(self.input_path,"Detections","High confidence")

        print('Moving images for review')
        for img in review:
            if os.path.exists(os.path.join(self.detections_images,img)):
                shutil.move(os.path.join(self.detections_images,img),os.path.join(detec,img))
        
    def img_mover(self):
        #make needed folders
        self.make_outputs()
        #obtain a list of names of images
        img_list = self.img_list()
        og_shapefile = os.path.join(self.shp_folder,'merge.shp')
        if not os.path.exists(og_shapefile):
            for root, dir, files in os.walk(self.shp_folder):
                for item in files:
                    if item.endswith(".shp"):
                        og_shapefile = os.path.join(root,item)
                        break
        unused_img = self.shp_reader(shapefile=og_shapefile,img_list=img_list)
        #move unuesd images
        self.move_unused(unused_img=unused_img)
        print('all done :)')

    def img_mover_conf(self):
        #make needed folders
        self.make_outputs()
        if not os.path.exists(os.path.join(self.input_path,"Detections","High confidence")):
            os.makedirs(os.path.join(self.input_path,"Detections","High confidence"))
        #obtain a list of names of images
        img_list = self.img_list()
        og_shapefile = os.path.join(self.shp_folder,'merge.shp')
        if not os.path.exists(og_shapefile):
            for root, dir, files in os.walk(self.shp_folder):
                for item in files:
                    if item.endswith(".shp"):
                        og_shapefile = os.path.join(root,item)
                        break

        unused_img,review = self.shp_reader_conf(og_shapefile,img_list)
        self.move_unused(unused_img)
        self.move_to_review(review)
        print('all done :)')


if __name__ == "__main__":
    # Defining the parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline', type=str, default=None, help='input path for pipeline folder')
    parser.add_argument('--confidence',type = float, default=0.0)
    opt = parser.parse_args()
    
    skip1 = ['Unzipped','R3P3ADD']
    in_folders = sorted(glob(opt.pipeline+"/*/", recursive = True))
    for fold in in_folders:
        route = os.path.basename(os.path.dirname(fold).split("\\")[-1])
        if route in skip1:
            print('skip',route)
            continue
        if check_outputs(fold) == False:
            print('missing folders', route)
            continue

        print('Working on ', route)
        image_finder = MissingPoints(input_path=fold,conf = opt.confidence)
        
        if opt.confidence == 0.0:
            # moving images
            image_finder.img_mover()
        else:
            image_finder.img_mover_conf()

