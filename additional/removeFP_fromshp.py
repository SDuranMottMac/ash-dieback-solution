import argparse
import os
from osgeo import osr
from osgeo import ogr
from osgeo import gdal
from glob import glob
import shutil
import csv
from tqdm import tqdm
import time

'''1)duplicate shapefile
2) check validity - does FP folder exist? skip if not
3) get list of images in False Positives
4) (FPTP) in attribute table - 
    a) add FP column
    b) if img in FP gets FP - TP gets TP
4) (remove) -
    a) remove rows in table if FP


5)move duplicates 
 '''
def find_csv(input_path):
    print('collecting csv')
    output = os.path.join(input_path,'Detections', 'output')
    csv_list = []
    for root,fold,files in os.walk(output):
        for item in files:
            if item.endswith('.csv'):
                csv_list.append(os.path.join(root,item))
    
    return csv_list

def height_band(MM_height):
    try:
        height = int(MM_height)
    except:
        return '25m+'
    if height < 3:
        return '0m-2m'
    elif height < 5:
        return '3m-4m'
    elif height < 10:
        return '5m-9m'
    elif height < 15:
        return '10m-14m'
    elif height <20:
        return '15m-19m'
    elif height < 25:
        return '20m-24m'
    else:
        return '25m+'

def h_band_conwy(MM_height):
    try:
        height = int(MM_height)
    except:
        return '15m+'
    if height < 5:
        return '0m-4m'
    elif height <10:
        return '5m-9m'
    elif height<15:
        return '10m-14m'
    else:
        return '15m+'

def ash_band(dieback):
    dieback = int(dieback)
    if dieback == 1:
        return '100-76'
    elif dieback == 2:
        return '75-51'
    elif dieback ==3:
        return '50-26'
    elif dieback == 4:
        return '25-0'

def detect_ang(img_name):
        if 'front' in img_name:
            return 'Front'
        return 'Rear'
         
def missing_img(img_name,TP_list):
        if img_name in TP_list:
            return 'Y'
        return 'N'

class RemoveShpPoints():
    #change to add attributes 
    def __init__(self,input_path) -> None:
        self.input_path =input_path
        self.shp_folder = os.path.join(input_path,"Shapefile")
        self.FP_folder = os.path.join(input_path,'Detections','False Positive')
        self.detections = os.path.join(input_path,'Detections')
        self.duplicates = os.path.join(input_path,'Detections','Duplicates')
    
    def check_outputs(self):
        if os.path.exists(self.shp_folder) and os.path.exists(self.FP_folder) and os.path.exists(os.path.join(self.input_path,'SS')):
            print('all folders good')
            return True

        return False

    def frontRear(self):
        #find if it is front or rear
        for root, fold, files in os.walk(self.FP_folder):
            for item in files:
                if item.endswith('.png'):
                    if 'rear' in item:
                        return 'rear'
                    else:
                        return 'front'

    def copy_shp(self):
        #copy all files from shp folder to shapefile folder if shapefile has not already been copied
        route = os.path.basename(self.input_path.split("\\")[-2])
        if os.path.exists((os.path.join(self.input_path,'SS','Shapefiles','og_shapefile'))) == False:
            print('Copying shapefile')
            fr = self.frontRear()
            #make shapefile folder for original shapefile
            os.makedirs(os.path.join(self.input_path,'SS','Shapefiles','og_shapefile'))
            for root, fold, files in os.walk(self.shp_folder):
                for item in files:
                    exten = item[-4:]
                    name = route + '-'+ fr + exten
                    #copy original with new name to new location
                    shutil.copy(os.path.join(root,item),os.path.join(self.input_path,'SS','Shapefiles','og_shapefile',name))
                    #rename copy to be modified
                    os.rename(os.path.join(root,item),os.path.join(root,name))

        #rename shapfile to proper name
        else:
            fr = self.frontRear()
            for root, fold, files in os.walk(self.shp_folder):
                for item in files:
                    exten = item[-4:]
                    name = route + '-'+ fr + exten
                    #rename copy to be modified
                    os.rename(os.path.join(root,item),os.path.join(root,name))


    
    def img_list(self):
        print('finding list of FP images')
        img_list = []
        for root, dirs, files in os.walk(self.FP_folder):
            for item in files:
                if item.endswith('.jpg') or item.endswith('.png'):
                    img_list.append(item)

        return img_list

    def TP_img(self):
        tp_list =[]
        for root, dirs, files in os.walk(self.detections):
            for item in files:
                basename = os.path.dirname(root)
                if basename == 'False Positive':
                    continue 
                if item.endswith('.jpg') or item.endswith('.png'):
                    tp_list.append(item)
        return tp_list
    
    def img_dup(self):
        #find list of duplicate images if there is a duplicate directory
        img_dup =[]
        if os.path.exists(self.duplicates):
            for root, dirs,files in os.walk(self.duplicates):
                for item in files:
                    if item.endswith('.jpg') or item.endswith('.png'):
                        img_dup.append(item)
        return img_dup

    def move_FP(self):
        FP_detec = os.path.join(self.input_path,'SS','False Positive','Detections')
        FP_nbbox = os.path.join(self.input_path,'SS','False Positive','No_bbox')
        nobbox = os.path.join(self.input_path,'No_bbox','False Positive')
        if not (os.path.exists(FP_detec) and os.path.exists(FP_nbbox)):
            os.makedirs(FP_detec)
            os.makedirs(FP_nbbox)
        #move all FP to respective folders
        img_list = self.img_list()
        for img in img_list:
            shutil.move(os.path.join(self.FP_folder,img),os.path.join(FP_detec,img))
            shutil.move(os.path.join(nobbox,img),os.path.join(FP_nbbox,img))
        
    def shp_remove(self,img_,TP_list):
        #obtain the shapefile
        for root,fold,files in os.walk(self.shp_folder):
            for item in files:
                if item.endswith('.shp'):
                    shapefile = os.path.join(root,item)
                
                #copy to no change
                if not os.path.exists(os.path.join(self.input_path,'SS','Shapefiles','no_change')):
                    os.makedirs(os.path.join(self.input_path,'SS','Shapefiles','no_change'))
                shutil.copy(os.path.join(root,item),os.path.join(self.input_path,'SS','Shapefiles','no_change',item))
        
        img_dup = self.img_dup()
        driver =ogr.GetDriverByName('ESRI Shapefile')
        table = driver.Open(shapefile,1)
        layer = table.GetLayer()
        #add new columns 
        height = ogr.FieldDefn("Height", ogr.OFTString)
        #height.SetWidth(10)
        layer.CreateField(height)

        dieback = ogr.FieldDefn('Canopy',ogr.OFTString)
        #dieback.SetWidth(15)
        layer.CreateField(dieback)

        ex_image = ogr.FieldDefn('Exists',ogr.OFTString)
        #ex_image.SetWidth(10)
        layer.CreateField(ex_image)

        angle = ogr.FieldDefn('Angle',ogr.OFTString)
        #angle.SetWidth(10)
        layer.CreateField(angle)


        #find paths in feature that coorespond with images
        #make attribute table changes here
        for feature in tqdm(layer,desc='Changing attribute table'):
            path = feature.GetField('Path')
            img_name = path.split('\\')
            img_name = img_name[-1]
            if img_name in img_ or img_name in img_dup:
                #delete feature if FP or duplicate
                layer.DeleteFeature(feature.GetFID())
                table.ExecuteSQL('REPACK ' + layer.GetName()) 
                continue
            
            #else leave in make table changes 
            #Height Band 
            layer.SetFeature(feature)
            h_band = height_band(feature.GetField('MM_Height'))
            feature.SetField('Height',h_band)
            layer.SetFeature(feature)
           
            #ash band
            layer.SetFeature(feature)
            a_band = ash_band(feature.GetField('Dieback'))
            feature.SetField('Canopy',a_band)
            layer.SetFeature(feature)

            #angle
            layer.SetFeature(feature)
            ang = detect_ang(img_name)
            feature.SetField('Angle',ang)
            layer.SetFeature(feature) 

            #existing image
            layer.SetFeature(feature)
            if img_name in TP_list:
                feature.SetField('Exists','Y')
            else:
                feature.SetField('Exists','N')
            layer.SetFeature(feature)

            time.sleep(1)
            table.ExecuteSQL('REPACK ' + layer.GetName()) 

        layer.ResetReading()
        time.sleep(0.01)

    def shp_FPTP(self, img_):
        #obtain the shapefile
        for root,fold,files in os.walk(self.shp_folder):
            for item in files:
                if item.endswith('.shp'):
                    shapefile = os.path.join(root,item)
                    break
            break
        
        driver =ogr.GetDriverByName('ESRI Shapefile')
        table = driver.Open(shapefile,1)
        layer = table.GetLayer()

        # Add new field on Att table (FP)
        FPTP = ogr.FieldDefn("FP/TP", ogr.OFTString)
        layer.CreateField(FPTP)

        #find paths in feature that coorespond with images
        for feature in layer:
            path = feature.GetField('Path')
            img_name = path.split('\\')
            img_name = img_name[-1]
            if img_name in img_:
                #set FP column to  FP if name is in list
                feature.SetField("FP/TP", 'FP')
                layer.SetFeature(feature)
            else:
                #set FPTP column to TP
                feature.SetField("FP/TP", 'TP')
                layer.SetFeature(feature)

        layer.ResetReading()
        
    def shp_changer(self):
        good = self.check_outputs()
        if not good:
            print('Missing an input folder')
            
        else:
            #copy original shp
            self.copy_shp()
            #obtain list of images in FP folder
            img_ = self.img_list()
            TP_list = self.TP_img()
            #annotate shapefile
            #self.shp_FPTP(img_=img_)
            self.shp_remove(img_=img_,TP_list=TP_list)
            self.move_FP()



class csv_FP():
    def __init__(self,input_path) -> None:
        self.input_path =input_path
        self.FP_folder = os.path.join(input_path,'Detections','False Positive')
        self.duplicates = os.path.join(input_path,'Detections','Duplicates')
        self.detections = os.path.join(input_path,'Detections')
        self.csv = find_csv(input_path)
    
    def make_outputs(self):
        print('Making False Postive folder')
        if os.path.exists(os.path.join(self.detections,'False Positive')):
            return
        else:
            os.makedirs(self.FP_folder)
            os.makedirs(self.duplicates)

    def check_outputs(self):
        if os.path.exists(os.path.join(self.detections,'output')) and os.path.exists(self.FP_folder):
            print('all folders good')
            return True

        print('output',os.path.exists(os.path.join(self.input_path,'output')))
        print('FP_folder',os.path.exists(self.FP_folder))
        return False
    
    def readMove_csv(self):
        #move all images in all csvs
        for path in self.csv:
            #read csv path
            print('Reading csv')
            with open(path) as fcsv:
                reader = csv.reader(fcsv)
                for row in reader:
                    img_name = row[0]
                    if len(row)==2: #only FP
                        #move image if it is in detections folder
                        if os.path.exists(os.path.join(self.detections,img_name)):
                            #remove if it is in FP folder already (should not happen, but in case of a duplicate, remove)
                            if os.path.exists(os.path.join(self.FP_folder,img_name)):
                                print('skip and removing')
                                os.remove(os.path.join(self.FP_folder,img_name))
                                continue
                            #move to FP
                            shutil.move(os.path.join(self.detections,img_name),os.path.join(self.FP_folder,img_name))
                    elif len(row)==3: #FP and duplicates
                        try:
                            label_FP = int(row[1])
                            label_dup = int(row[2])
                        except:
                            continue
                        if label_FP > label_dup:
                            #move to FP
                            if os.path.exists(os.path.join(self.detections,img_name)):
                            #remove if it is in FP folder already (should not happen, but in case of a duplicate, remove)
                                if os.path.exists(os.path.join(self.FP_folder,img_name)):
                                    print('skip and removing')
                                    os.remove(os.path.join(self.FP_folder,img_name))
                                    continue
                            #move to FP
                            shutil.move(os.path.join(self.detections,img_name),os.path.join(self.FP_folder,img_name))
                        elif label_dup > label_FP:
                            #move to duplicates
                            if os.path.exists(os.path.join(self.detections,img_name)):
                            #remove if it is in dup folder already (should not happen, but in case of a duplicate, remove)
                                if os.path.exists(os.path.join(self.FP_folder,img_name)):
                                    print('skip and removing')
                                    os.remove(os.path.join(self.FP_folder,img_name))
                                    continue
                                shutil.move(os.path.join(self.detections,img_name),os.path.join(self.duplicates,img_name))
                        

    def img_list(self,FP=True):
        if FP:
            folder = self.FP_folder
        else:
            folder = self.duplicates
        
        print('finding list of FP images')
        img_list = []
        for root, dirs, files in os.walk(folder):
            for item in files:
                if item.endswith('.jpg') or item.endswith('.png'):
                    img_list.append(item)
        return img_list

    def move_nobboxFP(self):
        print('moving no bounding box FP')
        FP_img = self.img_list()
        nbbox = os.path.join(self.input_path,"No_bbox")
        #make directory for false positive in no_bbox
        if os.path.exists(os.path.join(nbbox,'False Positive')):
            for img in FP_img:
                try:
                    shutil.move(os.path.join(nbbox,img),os.path.join(nbbox,'False Positive',img))
                except:
                    continue
        else:
            os.makedirs(os.path.join(nbbox,'False Positive'))
            for img in FP_img:
                try:
                    shutil.move(os.path.join(nbbox,img),os.path.join(nbbox,'False Positive',img))
                except:
                    continue

    def move_nobboxDup(self):
        print('moving no bounding box Duplicates')
        Dup_img = self.img_list(FP=False)
        nbbox = os.path.join(self.input_path,"No_bbox")
        if os.path.exists(os.path.join(nbbox,'Duplicates')):
            for img in Dup_img:
                try:
                    shutil.move(os.path.join(nbbox,img),os.path.join(nbbox,'Duplicates',img))
                except:
                    continue
        else:
            os.makedirs(os.path.join(nbbox,'Duplicates'))
            for img in Dup_img:
                try:
                    shutil.move(os.path.join(nbbox,img),os.path.join(nbbox,'Duplicates',img))
                except:
                    continue



    def run_(self):
        #make FP folder
        self.make_outputs()
        #check for output folder
        if not self.check_outputs():
            print('Missing an input folder')
        else:
            #make sure there is at least 1 csv
            if len(self.csv) ==0:
                print('no csv')
            else:
                #find FP and move in csv in detections
                self.readMove_csv()
                #find and move FP in nobbox
                self.move_nobboxFP()
                self.move_nobboxDup()
            
class clean():
    def __init__(self,input_path) -> None:
        self.input_path =input_path
        self.FP_folder = os.path.join(input_path,'Detections','False Positive')
        self.duplicates = os.path.join(input_path,'Detections','Duplicates')
        self.detections = os.path.join(input_path,'Detections')
        self.csv = find_csv(input_path)
    
    def namechange_csv(self):
        i=1
        for path in self.csv:
            l = path.split('\\')
            dir = l[-4]
            dir_p = os.path.dirname(path)
            newname = dir+'-'+str(i)+'.csv'
            os.rename(path,os.path.join(dir_p,newname))
            i+=1

    def move_csv(self):
        #make directory to put in
        if not os.path.exists(os.path.join(self.input_path,'SS','CSV')):
            os.makedirs(os.path.join(self.input_path,'SS','CSV'))
        
        newname_csv = find_csv(self.input_path)
        for path in newname_csv:
            name = os.path.basename(path)
            if name == 'Thumbs.db':
                continue
            shutil.move(path,os.path.join(self.input_path,'SS','CSV',name))
        

    def img_list(self,FP=True):
        if FP:
            folder = self.FP_folder
        else:
            folder = self.duplicates
        
        print('finding list of FP images')
        img_list = []
        for root, dirs, files in os.walk(folder):
            for item in files:
                if item.endswith('.jpg') or item.endswith('.png'):
                    img_list.append(item)
        return img_list

    def move_FP(self):
        FP_detec = os.path.join(self.input_path,'SS','False Positive','Detections')
        FP_nbbox = os.path.join(self.input_path,'SS','False Positive','No_bbox')
        nobbox = os.path.join(self.input_path,'No_bbox','False Positive')
        if not (os.path.exists(FP_detec) and os.path.exists(FP_nbbox)):
            os.makedirs(FP_detec)
            os.makedirs(FP_nbbox)
        #move all FP to respective folders
        img_list = self.img_list()
        for img in img_list:
            if img == 'Thumbs.db':
                continue
            shutil.move(os.path.join(self.FP_folder,img),os.path.join(FP_detec,img))
            shutil.move(os.path.join(nobbox,img),os.path.join(FP_nbbox,img))
        #remove empty FP folders
        
    
    def move_dup(self):
        if os.path.exists(os.path.join(self.input_path,'Detections','Duplicates')):
            dup_detec = os.path.join(self.input_path,'SS','Duplicates','Detections')
            dup_nbbox = os.path.join(self.input_path,'SS','Duplicates','No_bbox')
            nobbox = os.path.join(self.input_path,'No_bbox','Duplicates')
            if not (os.path.exists(dup_detec) and os.path.exists(dup_nbbox)):
                os.makedirs(dup_detec)
                os.makedirs(dup_nbbox)
            #move all dup to respective folders
            img_list = self.img_list(FP=False)
            if len(img_list)>0:
                for img in img_list:
                    if img == 'Thumbs.db':
                        continue
                    shutil.move(os.path.join(self.duplicates,img),os.path.join(dup_detec,img))
                    shutil.move(os.path.join(nobbox,img),os.path.join(dup_nbbox,img))
                
    
    def move_highconf(self):
        HC = os.path.join(self.detections,'High Confidence')
        if os.path.exists(HC):
            #move images to detections
            for root,fold,files in os.walk(HC):
                for item in files:
                    if item == 'Thumbs.db':
                        continue
                    shutil.move(os.path.join(root,item),os.path.join(self.detections,item))
            
    def rmv_direct(self):
        print('removing directories')
        delete_dict = {self.FP_folder:False,os.path.join(self.input_path,'No_bbox','False Positive'):False,self.duplicates:False,os.path.join(self.input_path,'No_bbox','Duplicates'):False,
        os.path.join(self.detections,'High Confidence'):False,os.path.join(self.detections,'output'):False}
        i = 0
        #will loop until all folders are deleted
        while sum(delete_dict.values()) < 6 or i >1000:
            print(sum(delete_dict.values()))
            if i%100 == 0:
                for key in delete_dict:
                    delete_dict[key]= not os.path.exists(key)
            
            for key in delete_dict:
    
                if delete_dict[key]==True:
                    continue
                try:
                    shutil.rmtree(key)
                    delete_dict[key]=True 
                except:
                    print(delete_dict.values())
                    continue
        i+=1

    def Run_clean(self):
        self.namechange_csv()
        self.move_csv()
        self.move_FP()
        self.move_dup()
        self.move_highconf()
        self.rmv_direct()

if __name__ == "__main__":
    # Defining the parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline', type=str, default=None, help='input path for pipeline folder, glob')
    opt = parser.parse_args()
    
    skip = ['Unzipped','ADDR1P1']

    in_folders = sorted(glob(opt.pipeline+"/*/", recursive = True))
    for fold in in_folders:
        route = os.path.basename(os.path.dirname(fold).split("\\")[-1])
        print('Working on', route)
        if route in skip:
            print('skipping')
            continue
        #csv reader and moves FP
        csv_ = csv_FP(input_path=fold)
        csv_.run_()
        #removes shapefile points
        pointrmv = RemoveShpPoints(input_path=fold)
        pointrmv.shp_changer()

        cleaner = clean(input_path=fold)
        cleaner.Run_clean()
    print('all done :)')
