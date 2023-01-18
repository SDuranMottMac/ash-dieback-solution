# importing libraries
# ===================
import argparse
import os
import json
from osgeo import osr
from osgeo import ogr
import pyproj
from glob import glob
from convertbng.util import convert_bng
import shutil
import numpy as np

# defining functionality
# =====================

def shapefile_from_dict(input_dict,out_directory,file_name):
    '''
    This function generates a shapefile with all the detections
    '''
    # first, we need to define a few variables
    shapepath = os.path.join(out_directory,file_name)

    # Define the Projection
    spatialReference = osr.SpatialReference()
    spatialReference.ImportFromProj4('+proj=utm +zone=30 +ellps=WGS84 +datum=WGS84 +units=m +no_defs')

    # Create a session in ArcGIS
    driver = ogr.GetDriverByName('ESRI Shapefile')
    shapeData = driver.CreateDataSource(shapepath)

    layer = shapeData.CreateLayer('layer1', spatialReference, ogr.wkbPoint)
    layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger)) # we need to create the ID field
    layer.CreateField(ogr.FieldDefn('MM_ID', ogr.OFTInteger))
    layer.CreateField(ogr.FieldDefn('Easting', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('Northing', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('MM_Height', ogr.OFTInteger))#we need to add a height field
    layer.CreateField(ogr.FieldDefn('Dieback', ogr.OFTInteger))
    layer.CreateField(ogr.FieldDefn('Path', ogr.OFTString))
    layer.CreateField(ogr.FieldDefn('Top_Height', ogr.OFTInteger))
    layer.CreateField(ogr.FieldDefn('Confidence', ogr.OFTReal))
    layerDefinition = layer.GetLayerDefn()

    point = ogr.Geometry(ogr.wkbPoint)

    # Generate the shapefile
    for ix,key in enumerate(input_dict):
        id_ = ix
        latitude = input_dict[key]["Latitude"]
        longitude = input_dict[key]["Longitude"]
        dieback = input_dict[key]["Dieback"]
        height = input_dict[key]["Height"]
        img_path = input_dict[key]["Image"]
        Top_h = int(input_dict[key]["Abs_height"])
        conf_det = input_dict[key]["Confidence"]

        #we need to transform the coordinates now
        p = pyproj.Proj(proj='utm', zone=30, ellps='WGS84')
        x,y = p(longitude, latitude)
        x_coord2 = float(x)
        y_coord2 = float(y)
        #we set the point
        point.SetPoint(0, x_coord2, y_coord2)
        feature = ogr.Feature(layerDefinition)
        feature.SetGeometry(point)
        # calculating easting and northing
        # try:
        #     easting = convert_bng(longitude,latitude)[0][0]
        #     northing = convert_bng(longitude,latitude)[1][0]
        # except:
        easting = 0.0
        northing = 0.0
        #we populate the corresponding info
        feature.SetFID(0)
        feature.SetField("id",id_)
        feature.SetField("MM_ID",id_)
        feature.SetField("Easting",easting)
        feature.SetField("Northing",northing)
        feature.SetField('MM_Height',height)
        feature.SetField('Dieback',dieback)
        feature.SetField('Path',img_path)
        feature.SetField('Top_Height',Top_h)
        feature.SetField('Confidence',conf_det)
        layer.CreateFeature(feature)

def merging_shp(input_dir,sub_folders):
    '''
    This function aims to merge different shapefiles into a same shapefile, so we can have all the files together.

    initial_shp - initial shapefile that will be copy and will form a base to which the rest shps will be appended
    merging_shp - list with the other shapefiles
    '''
    try:
        # creating the base shp to merge all the data
        folder_selected = sub_folders[0]
        filename = []
        # we get the name of all the files/subfolder in the folder
        sub_files = os.listdir(os.path.join(input_dir,folder_selected))
        # we append the absolute path of the .shp files
        for root,dirs,files in os.walk(os.path.join(input_dir,folder_selected)):
            for fle in files:
                if fle.endswith(".shp"):
                    filename.append(os.path.join(root,fle))
        
        shp_base = os.path.join(input_dir,folder_selected,filename[0])
        if not os.path.isdir(os.path.join(input_dir,"Shapefile")):
            os.makedirs(os.path.join(input_dir,"Shapefile"))
        out_name = os.path.join(input_dir,"Shapefile","merge.shp")

        cmd_line1 = "ogr2ogr -f \"ESRI Shapefile\" "+str(out_name)+" "+str(shp_base)
        os.system(cmd_line1)

        # appending all the other shps
        if (len(sub_folders) > 1):
            other_shp = []

            for item in sub_folders[1:]:
                sub_files = os.listdir(os.path.join(input_dir,item))
                filename = []
                for root,dirs,files in os.walk(os.path.join(input_dir,item)):
                    for fle in files:
                        if fle.endswith(".shp"):
                            filename.append(os.path.join(root,fle))

                shp_base = os.path.join(input_dir,item,filename[0])
                cmd_line2 = "ogr2ogr -f \"ESRI Shapefile\" -update -append "+str(out_name)+" "+str(shp_base)+" -nln merge"
                os.system(cmd_line2)
        
        return(True)
    except:
        return(False)
def change_path(path_,folder):
    abs_fold = folder
    img_name = path_.split("\\")[-1]
    return(os.path.join(abs_fold,img_name))

def generating_gdb(input_shp, img_folder,out_gdb):
    # importing specific libraries
    import arcpy
    # Obtaining the name
    gdb_name = os.path.basename(input_shp)[:-4]+".gdb"
    # Creating the geodatabase
    arcpy.management.CreateFileGDB(out_gdb,gdb_name,"CURRENT")
    # importing the shapefile
    arcpy.conversion.FeatureClassToFeatureClass(input_shp, os.path.join(out_gdb,gdb_name), "Detections")
    # enabling attachments
    arcpy.management.EnableAttachments(os.path.join(out_gdb,gdb_name,"Detections"))
    # modify the image paths
    # 1 - defining the code block:
    codeblock = """
def change_path(path_):
    abs_fold = """+repr(img_folder)+"""
    img_name = os.path.basename(path_)

    return(os.path.join(abs_fold,img_name))
    
    """   
    # 2 - changing the path
    arcpy.management.CalculateField(os.path.join(out_gdb,gdb_name,"Detections"), "Path", "change_path(!Path!)", "PYTHON3",codeblock)
    # creating a table for attaching images
    arcpy.management.GenerateAttachmentMatchTable(os.path.join(out_gdb,gdb_name,"Detections"),img_folder, os.path.join(out_gdb,gdb_name,"table_ash"), "Path", '', "RELATIVE")
    # adding attachments
    arcpy.management.AddAttachments(os.path.join(out_gdb,gdb_name,"Detections"), "OBJECTID", os.path.join(out_gdb,gdb_name,"table_ash"), "MatchID", "Filename", None)

class generating_shp_from_json():
    """
    This class process and generates shapefiles and gdbs
    """
    
    def __init__(self,input_path) -> None:
        self.input_path =input_path
        self.shp_folder = os.path.join(input_path,"Shapefile")
        self.detections_images = os.path.join(input_path,"Detections")
    
    def moving_unassigned(self):
        # grabbing all transformed files
        used_images = []
        # looping over all the json files generated
        for item in os.listdir(self.shp_folder):
            # if json
            if item.endswith("_det.json"):
                # read it and turn into dictionary
                json_f = os.path.join(self.shp_folder,item)
                with open(json_f) as json_file:
                    data_dict = json.load(json_file)
                # grabbing the images
                for key in data_dict:
                    image_name = os.path.basename(data_dict[key]["Image"])
                    used_images.append(image_name)
        # moving unused images
        os.makedirs(os.path.join(self.input_path,"SS","non-used-detections"))
        os.makedirs(os.path.join(self.input_path,"SS","non-used-nbbox"))
        for item in os.listdir(self.detections_images):
            if item not in used_images:
                shutil.move(os.path.join(self.input_path,"Detections",item),os.path.join(self.input_path,"SS","non-used-detections"))
                shutil.move(os.path.join(self.input_path,"No_bbox",item),os.path.join(self.input_path,"SS","non-used-nbbox"))
    
    def generating_shp(self):
        # grabbing all transformed files
        transformed_json = []
        # looping over all the json files generated
        for item in os.listdir(self.shp_folder):
            # if json
            if item.endswith("_det.json"):
                # read it and turn into dictionary
                json_f = os.path.join(self.shp_folder,item)
                with open(json_f) as json_file:
                    data_dict = json.load(json_file)
                # turn the dictionary into shp
                shapefile_from_dict(input_dict=data_dict,out_directory=self.shp_folder,file_name=item[:-4]+".shp")
                # appending the file
                transformed_json.append(os.path.join(self.shp_folder,item))
        # moving the json files to a new folder
        os.makedirs(os.path.join(self.shp_folder,"SS"),exist_ok=True)
        # moving the json to the folder
        for item in transformed_json:
            shutil.move(item,os.path.join(self.shp_folder,"SS"))
        
    def merging_shp(self):
        # looping over all the shp files generated
        to_be_merged = []
        for item in os.listdir(self.shp_folder):
            # if json
            if item.endswith(".shp"):
                if item != "merge.shp":
                    to_be_merged.append(os.path.join(self.shp_folder,item))
        if (len(to_be_merged) > 0):
            # defining the out name
            out_name = os.path.join(self.shp_folder,"merge.shp")
            # instantiating the merge shapefile with the first shp
            shp_base = to_be_merged[0]
            cmd_line1 = "ogr2ogr -f \"ESRI Shapefile\" "+str(out_name)+" "+str(shp_base)
            os.system(cmd_line1)

            # appending all the other shps
            if (len(to_be_merged) > 1):
                other_shp = []

                for item in to_be_merged[1:]:
                    shp_base = item
                    cmd_line2 = "ogr2ogr -f \"ESRI Shapefile\" -update -append "+str(out_name)+" "+str(shp_base)+" -nln merge"
                    os.system(cmd_line2)
        # moving the previous json to a SS folder
        if not os.path.isdir(os.path.join(self.shp_folder,"SS")):
            os.makedirs(os.path.join(self.shp_folder,"SS"))
        if len(to_be_merged) > 0:
            for item in to_be_merged:
                shutil.move(item,os.path.join(self.shp_folder,"SS"))
                shutil.move(item[:-4]+".dbf",os.path.join(self.shp_folder,"SS"))
                shutil.move(item[:-4]+".prj",os.path.join(self.shp_folder,"SS"))
                shutil.move(item[:-4]+".shx",os.path.join(self.shp_folder,"SS"))

    def remove_intermediate_shp(self):
        to_be_removed = []
        for item in os.listdir(self.shp_folder):
            # if json
            if item.endswith(".shp"):
                if item != "merge.shp":
                    to_be_removed.append(os.path.join(self.shp_folder,item))
        for item in to_be_removed:
            os.remove(os.path.join(self.shp_folder,item))

    def postprocessing_shp():
        pass
    def generating_gdb(self):
        # importing specific libraries
        import arcpy
        # Obtaining the name
        gdb_name = os.path.basename(os.path.join(self.shp_folder,"merge.shp"))[:-4]+".gdb"
        # Creating the geodatabase
        arcpy.management.CreateFileGDB(self.shp_folder,gdb_name,"CURRENT")
        # importing the shapefile
        arcpy.conversion.FeatureClassToFeatureClass(os.path.join(self.shp_folder,"merge.shp"), os.path.join(self.shp_folder,gdb_name), "Detections")
        # enabling attachments
        arcpy.management.EnableAttachments(os.path.join(self.shp_folder,gdb_name,"Detections"))
        # modify the image paths
        # 1 - defining the code block:
        codeblock = """
    def change_path(path_):
        abs_fold = """+repr(self.detections_images)+"""
        img_name = os.path.basename(path_)

        return(os.path.join(abs_fold,img_name))
        
        """   
        # 2 - changing the path
        arcpy.management.CalculateField(os.path.join(self.shp_folder,gdb_name,"Detections"), "Path", "change_path(!Path!)", "PYTHON3",codeblock)
        # creating a table for attaching images
        arcpy.management.GenerateAttachmentMatchTable(os.path.join(self.shp_folder,gdb_name,"Detections"),self.detections_images, os.path.join(self.shp_folder,gdb_name,"table_ash"), "Path", '', "RELATIVE")
        # adding attachments
        arcpy.management.AddAttachments(os.path.join(self.shp_folder,gdb_name,"Detections"), "OBJECTID", os.path.join(self.shp_folder,gdb_name,"table_ash"), "MatchID", "Filename", None)


class ReviewImagesGenerator():

    def __init__(self,front_path,rear_path) -> None:
        self.front_path =front_path
        self.rear_path =rear_path
    
    def list_healthy(self):

        front_healthy = []
        rear_healthy = []

        # looping over all the json files generated
        os.makedirs(os.path.join(self.front_path,"to_Review"))
        for item in os.listdir(os.path.join(self.front_path,"Shapefile","SS")):
            # if json
            if item.endswith("_det.json"):
                # read it and turn into dictionary
                json_f = os.path.join(self.front_path,"Shapefile","SS",item)
                with open(json_f) as json_file:
                    data_dict = json.load(json_file)
                # grabbing the images
                for key in data_dict:
                    image_name = os.path.basename(data_dict[key]["Image"])
                    if data_dict[key]["Dieback"] < 2:
                        shutil.copy(os.path.join(self.front_path,"Detections",image_name),os.path.join(self.front_path,"to_Review"))
                        front_healthy.append(image_name)

        # looping over all the json files generated
        os.makedirs(os.path.join(self.rear_path,"to_Review"))
        for item in os.listdir(os.path.join(self.rear_path,"Shapefile","SS")):
            # if json
            if item.endswith("_det.json"):
                # read it and turn into dictionary
                json_f = os.path.join(self.rear_path,"Shapefile","SS",item)
                with open(json_f) as json_file:
                    data_dict = json.load(json_file)
                # grabbing the images
                for key in data_dict:
                    image_name = os.path.basename(data_dict[key]["Image"])
                    if data_dict[key]["Dieback"] < 2:
                        shutil.copy(os.path.join(self.rear_path,"Detections",image_name),os.path.join(self.rear_path,"to_Review"))
                        rear_healthy.append(image_name)
        return(front_healthy,rear_healthy)
        
    def haversine_np(self,lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance between two points 
        on the earth (specified in decimal degrees)
        """
        # convert decimal degrees to radians 
        lon1 = np.deg2rad(np.array(lon1))
        lat1 = np.deg2rad(np.array(lat1))
        lon2 = np.deg2rad(np.array(lon2))
        lat2 = np.deg2rad(np.array(lat2))

        # haversine formula 
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = np.sin(abs(dlat)/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(abs(dlon)/2)**2
        c = 2 * np.arcsin(np.sqrt(a)) 
        # Radius of earth in kilometers is 6371
        # km = 6371* c
        km = 6372.8*c
        m = km*1000
        return(m)

    def obtaining_duplicates_rear(self,front_img,rear_imgs):
       
        # turning images to lat long
        front_lat = list()
        front_long = list()
        rear_lat = list()
        rear_long = list()

        for item in front_img:
            lat = item.split("(")[1].split(",")[0]
            long = item.split("(")[1].split(",")[1].split(")")[0]
            front_lat.append(float(lat))
            front_long.append(float(long))
        for item in rear_imgs:
            lat = item.split("(")[1].split(",")[0]
            long = item.split("(")[1].split(",")[1].split(")")[0]
            rear_lat.append(float(lat))
            rear_long.append(float(long))
        
        duplicate_rear = []
        os.makedirs(os.path.join(self.rear_path,"SS","duplicates"))
        for ix,item in enumerate(rear_imgs):
            print("Dealing with file "+str(item))

            # perform the calculations
            lon1 = np.array(front_long)
            lat1 = np.array(front_lat)
            lon2 = np.array(rear_long[ix])
            lat2 = np.array(rear_lat[ix])
            sep = self.haversine_np(lon1,lat1,lon2,lat2)
            eval_matrix = np.any(sep < 2)
            # evaluating whether any previous point is less than 15 meters
            if eval_matrix:
                duplicate_rear.append(item)
        
        for item in duplicate_rear:
            shutil.move(os.path.join(self.rear_path,"to_Review",item),os.path.join(self.rear_path,"SS","duplicates"))
                

if __name__ == "__main__":

    option = input("Do you need to obtain shapefile [1] or images to review [2]? [1/2]")
    if option == "1":
        # Defining the parse arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--main_folder', type=str, default=None, help='optional svo file or folder')
        opt = parser.parse_args()


        shp_processer = generating_shp_from_json(input_path=opt.main_folder)
        print("Proceeding to move unassigned images")
        shp_processer.moving_unassigned()
        print("Done!")
        print("Proceeding to generate shp")
        shp_processer.generating_shp()
        print("...and now merging them")
        shp_processer.merging_shp()
        print("Done!")
    
    else:
        # Defining the parse arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--front_path', type=str, default=None, help='optional svo file or folder')
        parser.add_argument('--rear_path', type=str, default=None, help='optional svo file or folder')
        opt = parser.parse_args()

        Image_generator = ReviewImagesGenerator(opt.front_path, opt.rear_path)
        # moving healthy images
        print("Moving healthy images")
        front_img,rear_imgs = Image_generator.list_healthy()
        # obtaining duplicates
        print("Moving duplicates away")
        Image_generator.obtaining_duplicates_rear(front_img,rear_imgs)









