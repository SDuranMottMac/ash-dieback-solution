import argparse
import os
from osgeo import osr
from osgeo import ogr
from osgeo import gdal
from glob import glob
import shutil
from tqdm import tqdm
import time
## add duplication folders
## add no change shapefiles


def find_shp(path):
    for root,fold,files in os.walk(path):
        for item in files:
            if item.endswith('.shp'):
                return os.path.join(root,item)

class combine():
    def __init__(self,input_path,skip) -> None:
        self.input_path = input_path
        
        self.shapefile_Fin = os.path.join(input_path,'Final','Shapefile')
        self.detections_Fin = os.path.join(input_path,'Final','Detections')
        self.nbbox_Fin = os.path.join(input_path,'Final','No_bbox')
        self.monitoring_fin = os.path.join(input_path,'Final','Monitoring')
        self.SS_Fin = os.path.join(input_path,'Final','SS')

        self.skip = skip
        self.glob_ = sorted(glob(input_path+"/*/", recursive = True))
        self.client = os.path.basename(input_path)

    def makeglob(self):
        for item in self.skip:
            if os.path.join(self.input_path,item) in self.glob_:
                self.glob_.remove(os.path.join(self.input_path,item))

    def make_outputs(self):
        #make outputs if they do not exist
        output_fold = [self.shapefile_Fin,self.detections_Fin,self.nbbox_Fin,self.monitoring_fin,self.SS_Fin,self.shapefile_Fin]
        for path in output_fold:
            if os.path.exists(path):
                continue
            os.makedirs(path)

    def merge_monitoring(self):
        if os.path.exists(os.path.join(self.monitoring_fin,'Images')) == False:
            os.makedirs((os.path.join(self.monitoring_fin,'Images')))
            os.makedirs(os.path.join(self.monitoring_fin,'Labels'))

        for path in tqdm(self.glob_,desc='Merging Monitoring'):
            monit = os.path.join(path,'Monitoring')
            pglob = sorted(glob(monit+"/*/", recursive = True))
            for path in pglob:
                for root,fold,files in os.walk(path):
                    for item in files:
                        if 'Labels' in root:
                            shutil.move(os.path.join(root,item),os.path.join(self.monitoring_fin,'Labels',item))
                        else:
                            
                            shutil.move(os.path.join(root,item),os.path.join(self.monitoring_fin,'Images',item))
    
    def merge_SS(self):
        FP = os.path.join(self.SS_Fin,'False Positive')
        D = os.path.join(self.SS_Fin,'non-used-detections')
        NB = os.path.join(self.SS_Fin,'non-used-nbbox')
        SHP = os.path.join(self.SS_Fin,'Shapefiles')
        OUT = os.path.join(self.SS_Fin,'CSV')
        DUP = os.path.join(self.SS_Fin,'Duplicates')
        
        if os.path.exists(FP) == False:
            os.makedirs(os.path.join(FP,'Detections'))
            os.makedirs(os.path.join(FP,'No_bbox'))
            os.makedirs(D)
            os.makedirs(NB)
            os.makedirs(SHP)
            os.makedirs(OUT)
            os.makedirs(os.path.join(DUP,'Detections'))
            os.makedirs(os.path.join(DUP,'No_bbox'))

        for path in tqdm(self.glob_,desc='Merging SS folders'):
            SS = os.path.join(path,'SS')
            
            #move non used detections and no_bbox
            detect = os.path.join(SS, 'non-used-detections')
            nbbox = os.path.join(SS, 'non-used-nbbox')
            for root,fold,files in os.walk(detect):
                for item in files:
                    #move both detect and nobbox as item has same name
                    shutil.move(os.path.join(root,item),os.path.join(D,item))
                    shutil.move(os.path.join(nbbox,item),os.path.join(NB,item))
            
            #move FP 
            FP_detect = os.path.join(SS,'False Positive','Detections')
            FP_nobbox = os.path.join(SS,'False Positive','No_bbox')

            for root,fold,files in os.walk(FP_detect):
                for item in files:
                    shutil.move(os.path.join(root,item),os.path.join(FP,'Detections'))
                    shutil.move(os.path.join(FP_nobbox,item),os.path.join(FP,'No_bbox'))

            #move Duplicates
            dup_detect= os.path.join(SS,'Duplicates','Detections')
            dup_nobbox = os.path.join(SS,'Duplicates','No_bbox')

            for root,fold,files in os.walk(dup_detect):
                for item in files:
                    shutil.move(os.path.join(root,item),os.path.join(DUP,'Detections',item))
                    shutil.move(os.path.join(dup_nobbox,item),os.path.join(DUP,'No_bbox',item))
            
            #move output
            for root,fold,files in os.walk(os.path.join(SS,'CSV')):
                for item in files:
                    shutil.move(os.path.join(root,item),os.path.join(OUT,item))

    
    def merge_detect_nobbox(self):

        for path in tqdm(self.glob_,desc='Merging Detections and No_bbox'):
            detect = os.path.join(path, 'Detections')
            nbbox = os.path.join(path, 'No_bbox')
            #check for confidence folder
            if os.path.exists(os.path.join(path,'Detections','High confidence')):
                for root,fold,files in os.walk(os.path.join(path,'Detections','High confidence')):
                    for item in files:
                        shutil.move(os.path.join(root,item),os.path.join(path,"Detections"))

            #move images
            for root,fold,files in os.walk(detect):
                for item in files:
                    if item.endswith('.png'):
                        shutil.move(os.path.join(root,item),os.path.join(self.detections_Fin,item))
                        shutil.move(os.path.join(nbbox,item),os.path.join(self.nbbox_Fin,item))

        #leave folders for now
    
    def change_shp_name(self,path):
        shp_name = self.client +'-merge_all'
        for root,fold,files in os.walk(path):
            for item in files:
                if 'merge' in item:
                    exten = item[-4:]
                    os.rename(os.path.join(root,item),os.path.join(root,shp_name+exten))

    def merge_shp(self):
        #merge both og and edited ones
        if os.path.exists(os.path.join(self.SS_Fin,'Shapefiles','og_shapefile')) == False:
            os.makedirs(os.path.join(self.SS_Fin,'Shapefiles','Seperate_Shapefiles'))
            os.makedirs(os.path.join(self.SS_Fin,'Shapefiles','og_shapefile'))
            os.makedirs(os.path.join(self.SS_Fin,'Shapefiles','FrontRear_merged'))
            os.makedirs(os.path.join(self.SS_Fin,'Shapefiles','no_change'))
        
        i = 0
        for path in tqdm(self.glob_,desc='Merging Shapefiles'):
            shapefile_p = os.path.join(path,'Shapefile')
            p_og_shp = os.path.join(path,'SS','Shapefiles','og_shapefile')
            p_sep_shp = os.path.join(path,'SS','Shapefiles','Seperate_Shapefiles')
            p_no_change = os.path.join(path,'SS','Shapefiles','no_change')

            og_shp = find_shp(p_og_shp)
            shapefile_shp = find_shp(shapefile_p)
            
            for root, fold, files in os.walk(shapefile_p):
                for item in files:
                    #copy front/rear seperately
                    shutil.copy(os.path.join(root,item),os.path.join(self.SS_Fin,'Shapefiles','FrontRear_merged',item))
            
            for root,fold,files in os.walk(p_sep_shp):
                for item in files:
                    #move all seperated routes 
                    shutil.move(os.path.join(root,item),os.path.join(self.SS_Fin,'Shapefiles','Seperate_Shapefiles',item))
            
            for root,fold,file in os.walk(p_no_change):
                for item in files:
                    #move no change and keep individual
                    shutil.move(os.path.join(root,item),os.path.join(self.SS_Fin,'Shapefiles','no_change',item))
            
                
            #merge og shapefiles
            if i == 0:
                #if it is the first shapefile, we need to make the shapefiles to merge to 
                #make base shp to merge to 
                #SS
                out_name_SS = os.path.join(self.SS_Fin,'Shapefiles','og_shapefile','merge.shp')
                cmd_line1a = "ogr2ogr -f \"ESRI Shapefile\" "+str(out_name_SS)+" "+str(og_shp)
                os.system(cmd_line1a)

                #shapefile
                out_name_shapefile = os.path.join(self.shapefile_Fin,'merge.shp')
                cmd_line1b = "ogr2ogr -f \"ESRI Shapefile\" "+str(out_name_shapefile)+" "+str(shapefile_shp)
                os.system(cmd_line1b)

                i += 1
                continue

            #merge all others
            #SS
            out_name_SS = os.path.join(self.SS_Fin,'Shapefiles','og_shapefile','merge.shp')
            cmd_line2a = "ogr2ogr -f \"ESRI Shapefile\" -update -append "+str(out_name_SS)+" "+str(og_shp)+" -nln merge"
            os.system(cmd_line2a)

            #shapefile
            out_name_shapefile = os.path.join(self.shapefile_Fin,'merge.shp')
            cmd_line2b = "ogr2ogr -f \"ESRI Shapefile\" -update -append "+str(out_name_shapefile)+" "+str(shapefile_shp)+" -nln merge"
            os.system(cmd_line2b)

        #change merge shp names
        self.change_shp_name(os.path.join(self.SS_Fin,'Shapefiles','og_shapefile'))
        self.change_shp_name(os.path.join(self.shapefile_Fin))


    def Run(self):
        self.makeglob()
        #print('Making output folders')
        self.make_outputs()
        #print('Merging monitoring')
        #self.merge_monitoring()
        #print('Merging SS folder')
        #self.merge_SS()
        #print('Merging Detections and No_bbox folders')
        #self.merge_detect_nobbox()
        #print('Merging shapefiles')
        self.merge_shp()
