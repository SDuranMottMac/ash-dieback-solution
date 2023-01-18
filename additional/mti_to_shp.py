# importing libraries
from genericpath import isfile
import os
from osgeo import osr
from osgeo import ogr
import pyproj
from glob import glob
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import math
from tqdm import tqdm
import pathlib
# import geopandas as gpd

# importing the duplicate module 
file_path = os.path.dirname( __file__)
os.chdir(file_path)
from duplicates_videos import DuplicateID

# adding functionality
# adding extra functionality
def haversine_np(lon1, lat1, lon2, lat2):
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
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    # Radius of earth in kilometers is 6371
    # km = 6371* c
    km = 6372.8*c
    m = km*1000
    return(m)

def getList(dictionary):
    return dictionary.keys()

def generating_timestamp(nano,year,month,day,hour,minute,second):
    nano = int(nano/1000)
    dt = datetime(int(year),int(month),int(day),int(hour),int(minute),int(second),nano,tzinfo=timezone.utc)
    epoch = datetime(1970,1,1,0,0,0,0,tzinfo=timezone.utc)
    timestamp = (dt-epoch)//timedelta(microseconds=1)
    timestamp =round(timestamp/1000000,2)
    return(timestamp)

def interpolating_df(input_dict):
    # first we need to transform the dictionary into a dataframe
    df = pd.DataFrame.from_dict(input_dict,orient="index")
    # now we are going to get the interpolated index
    idxs = list(df.index.values)
    starting = float(min(idxs))
    ending = float(max(idxs))
    print("max value is "+str(ending))
    print("min value is "+str(starting))
    assert (ending > starting)
    new_index = []
    new_value = starting

    indexing = range(int(starting*100),int(ending*100+1),1)
    for item in indexing:
        new_index.append(round(item/100,2))
    
    # now we can create the interpolated dictionary
    interpolated_df = pd.DataFrame(index=new_index,columns=["latitude","longitude","Altitude"])
    df_merged = pd.concat([df,interpolated_df],join="outer")

    df_merged = df_merged[~df_merged.index.duplicated(keep='first')]
    df_merged.sort_index(inplace=True)
    for col in df_merged:
       df_merged[col] =pd.to_numeric(df_merged[col],errors="coerce") 
    
    interpolated_merged = df_merged.interpolate(method="index")
    
    interpolated_dict = interpolated_merged.to_dict("index")
    #print(interpolated_dict)
    return(interpolated_dict)

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
    layerDefinition = layer.GetLayerDefn()

    point = ogr.Geometry(ogr.wkbPoint)

    # Generate the shapefile
    id_ = 0
    for key in input_dict:
        id_ = id_
        latitude = input_dict[key]["latitude"]
        longitude = input_dict[key]["longitude"]

        #we need to transform the coordinates now
        p = pyproj.Proj(proj='utm', zone=30, ellps='WGS84')
        x,y = p(longitude, latitude)
        x_coord2 = float(x)
        y_coord2 = float(y)
        #we set the point
        point.SetPoint(0, x_coord2, y_coord2)
        feature = ogr.Feature(layerDefinition)
        feature.SetGeometry(point)
        #we populate the corresponding info
        feature.SetFID(0)

        feature.SetField("id",id_)
        layer.CreateFeature(feature)
        id_ += 1

class reducing_shp_density():
    """
    This class reduces the .shp density to be managable
    """
    def __init__(self, input_dir, output_dir,shp):
        # defining input and output folder
        self.input_folder = input_dir
        self.output_folder = output_dir 
        self.shapename = shp
    
    def listing_directories(self):

        all_folders = glob(self.input_folder+"/*/", recursive = True)
        return(all_folders)
    
    def listing_all_shpdata(self):
        # Creating an empty list
        shp_data = []
        # Populating the list with all the svi videos
        for path, subdirs, files in os.walk(self.input_folder):
            for name in files:
                if name == self.shapename+".shp":
                    if os.path.join(path, name) not in shp_data:
                        shp_data.append(os.path.join(path, name))
        return(shp_data)
    
    def reading_shp(self, shp_path):
        file = ogr.Open(shp_path)
        shape = file.GetLayer(0)
        #first feature of the shapefile
        feature = shape.GetFeature()
        print(feature)
        first = feature.ExportToJson()
        print(first)
    
    def reading_shp2(self, shp_path):
        geo_shapefile = gpd.read_file(shp_path)
        return(geo_shapefile)

    def downsample_shp(self,shp_data):
        for item in shp_data:
            print("Dealing with file: "+str(item))
            # getting the directory of the file
            dirname = os.path.dirname(item)
            if os.path.isfile(os.path.join(dirname,"route_covered_downsampled.shp")):
                continue
            # reading the file
            geoDF = self.reading_shp2(item)
            # obtaining max and min index
            try:
                max_index =int(geoDF.max()["id"])
                min_index = 0
            except:
                continue
            # assessing which index must be dropped
            index_to_drop = []
            for i in range(max_index-min_index):
                if i % 1500 == 0:
                    continue
                else:
                    index_to_drop.append(i)
            # dropping index
            new_geoDF = geoDF.drop(index_to_drop,axis=0)
            # save the GeoDataFrame
            new_geoDF.to_file(filename= os.path.join(dirname,"route_covered_downsampled.shp"))
            

class PlottingMTData():
    """
    Creating the .shp for the mt kit data
    """

    def __init__(self, input_dir, output_dir):
        # defining input and output folder
        self.input_folder = input_dir
        self.output_folder = output_dir 
    
    def listing_directories(self):

        all_folders = glob(self.input_folder+"/*/", recursive = True)
        return(all_folders)
    
    def listing_all_mtdata(self, folder):
        # Creating an empty list
        mt_data = []
        # Populating the list with all the svi videos
        for path, subdirs, files in os.walk(folder):
            for name in files:
                if name.endswith(".txt"):
                    if os.path.join(path, name) not in mt_data:
                        mt_data.append(os.path.join(path, name))
        return(mt_data)
    
    def strip_mt_data(self,mt_file):
        # obtaining the file name
        file_name = os.path.basename(mt_file)
        # obtaining the parent directory
        purepath_obj = pathlib.PurePath(mt_file)
        file_directory = purepath_obj.parent
        # creating a list to write the new file
        file_to_write = []
        # defining new file name
        name = file_name.split(".")[0] + "-000.txt"
        # reading lines and only adding correctly formatted lines
        with open(mt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("//"):
                    pass
                else:
                    file_to_write.append(line)
        # writing a new file with only correctly formatted lines
        with open(os.path.join(file_directory,name), 'w+') as p:
            p.writelines(file_to_write)
        # removing old file - badly formatted
        os.remove(mt_file)
        # renaming the new files with the name of the older file
        os.rename(os.path.join(file_directory,name),mt_file)
    
    def split_mt_data(self, list_mtdata):
        # generating a list with files to be removed - those that exceed 500MB
        files_to_remove = []
        for file__ in tqdm(list_mtdata):
            # obtaining file name
            file_name = os.path.basename(file__)
            # obtaining parent directory path
            purepath_obj = pathlib.PurePath(file__)
            file_directory = purepath_obj.parent
            # obtaining file size
            filesize = os.path.getsize(file__)
            # checking whether it meets both conditions
            if ((filesize > 500000000) and (file_name.endswith(".txt"))):# 500MB
                #split it
                with open(file__, 'r') as f:
                    lines = f.readlines()
                    header = lines[0]
                    amnt = int(filesize/500000000)
                    split_files = np.array_split(lines[1:], amnt)

                    for idx, val in enumerate(split_files):
                        with open(os.path.join(file_directory,(file_name.split(".")[0]+"00{}.txt".format(idx))), 'w+') as p:
                            p.write(header)
                            p.writelines(val)
                
                # append it to the list
                files_to_remove.append(file__)
        
        # removing files too large
        for item in files_to_remove:
            os.remove(item)
    
    def mt_processing(self,mt_list):
        print("Post-processing the mt data")
        # create a dictionary to store all data
        mit_data = dict()
        # loop over all the items in the list
        for item in mt_list:
            try:
                # read the file
                print("reading file "+str(item))
                df_raw = pd.read_csv(item)
                print("Iterating over file")
                # Iterate over the data in the mt kit file        
                for index,row in df_raw.iterrows():
                    # we generate the timestamp
                    timestamp = generating_timestamp(nano=row['UTC_Nano'],year=row['UTC_Year'],month=row['UTC_Month'],day=row['UTC_Day'],hour=row['UTC_Hour'],minute=row['UTC_Minute'],second=row['UTC_Second'])
                    # add the timestamp as key to the dict
                    if timestamp in mit_data:
                        if ((math.isnan(mit_data[timestamp]["latitude"])) and (not math.isnan(row['Latitude']))):
                            mit_data[timestamp] = {"latitude":row['Latitude'],"longitude":row['Longitude'],"Altitude":row['Altitude']}
                            #mit_data[timestamp] = {"latitude":row['Latitude'],"longitude":row['Longitude'],"Altitude":row['Altitude']}
                        else:
                            continue
                    else:
                        mit_data[timestamp] = {"latitude":row['Latitude'],"longitude":row['Longitude'],"Altitude":row['Altitude']}
                        #mit_data[timestamp] = {"latitude":row['Latitude'],"longitude":row['Longitude'],"Altitude":row['Altitude']}
            except:
                try:
                    # read the file
                    print("Trying again to read the file")
                    df_raw = pd.read_csv(item,sep=",")
                    print("Read")
                    # Iterate over the data in the mt kit file        
                    for index,row in df_raw.iterrows():
                        # we generate the timestamp
                        timestamp = generating_timestamp(nano=row['UTC_Nano'],year=row['UTC_Year'],month=row['UTC_Month'],day=row['UTC_Day'],hour=row['UTC_Hour'],minute=row['UTC_Minute'],second=row['UTC_Second'])
                        # add the timestamp as key to the dict
                        if timestamp in mit_data:
                            if ((math.isnan(mit_data[timestamp]["latitude"])) and (not math.isnan(row['Latitude']))):
                                #mit_data[timestamp] = {"latitude":row['Latitude'],"longitude":row['Longitude'],"Altitude":row['Altitude'],"Pitch":row['Pitch'],"Roll":row['Roll']}
                                mit_data[timestamp] = {"latitude":row['Latitude'],"longitude":row['Longitude'],"Altitude":row['Altitude']}
                            else:  
                                continue
                        else:
                            #mit_data[timestamp] = {"latitude":row['Latitude'],"longitude":row['Longitude'],"Altitude":row['Altitude'],"Pitch":row['Pitch'],"Roll":row['Roll']}
                            mit_data[timestamp] = {"latitude":row['Latitude'],"longitude":row['Longitude'],"Altitude":row['Altitude']}
                except Exception as e:
                    print(e)

        # Get a list with all the timestamp collected                      
        all_keys = getList(mit_data)
        del_keys = []
        # loop over all the timestamp to see if there are empty ones
        for key in all_keys:
            if math.isnan(mit_data[key]["latitude"]):
                del_keys.append(key)
        # Remove those timestamp not containing info
        for key in del_keys:
            del mit_data[key]
        # Interpolating the final timestamp
        mit_data= interpolating_df(input_dict=mit_data)
        return(mit_data)
    
    def plotting_shp_no_duplicates(self):
        skipping_folder = ["One touch (J)"]
        print("Listing all directories")
        all_folders = self.listing_directories()
        print(all_folders)
        for fold in all_folders:
            print("Folder is "+fold)
            # obtaining the output directories
            directory_name = os.path.dirname(fold).split("\\")[-1]
            print("dealing with folder: "+str(directory_name))
            out_folder = os.path.join(self.output_folder,directory_name)
            if not os.path.isdir(out_folder):
                os.makedirs(out_folder)
            # checking whether it h as already been processed
            if os.path.isfile(os.path.join(out_folder,"MT_data.shp")):
                continue
            #cheking whether this is in the to-be-skipped folders
            if directory_name in skipping_folder:
                continue
            # listing all the mt data in the folder
            try:
                print("Listing mt data")
                mt_data = self.listing_all_mtdata(fold)
                print("Pre-processing the mt data")
                # preprocessing the mtdata
                for item in mt_data:
                    self.strip_mt_data(item)
                mt_data = self.listing_all_mtdata(fold)
                self.split_mt_data(mt_data)
                # generating the mt dic
                print("Processing the mt data")
                mt_data = self.listing_all_mtdata(fold)
                mt_dic = self.mt_processing(mt_data)
                # obtaining the list of duplicates
                print("Obtaining the duplicates")
                route1 = DuplicateID(input_folder=fold,output_folder=out_folder)
                dup,lat,long = route1.obtaining_duplicatesTS_Lat_Long()
                print("Removing duplicates")
                # removing those keys from dict
                for dd in dup:
                    if dd in mt_dic:
                        del mt_dic[dd]
                # plotting the shp
                print("Generating the shapefiles")
                shapefile_from_dict(input_dict=mt_dic,out_directory=out_folder,file_name="MT_data.shp")
            except Exception as e:
                print(e)
                print("Unable to process that folder")
                continue
    
    def plotting_shp(self):
        skipping_folder = ["One touch (J)"]
        all_folders = self.listing_directories()
        for fold in all_folders:
            # obtaining the output directories
            directory_name = os.path.dirname(fold).split("\\")[-1]
            print("dealing with folder: "+str(directory_name))
            out_folder = os.path.join(self.output_folder,directory_name)
            if not os.path.isdir(out_folder):
                os.makedirs(out_folder)
            # checking whether it h as already been processed
            if os.path.isfile(os.path.join(out_folder,"MT_data.shp")):
                continue
            #cheking whether this is in the to-be-skipped folders
            if directory_name in skipping_folder:
                continue
            # listing all the mt data in the folder
            try:
                print("Listing mt data")
                mt_data = self.listing_all_mtdata(fold)
                print("Pre-processing the mt data")
                # preprocessing the mtdata
                for item in mt_data:
                    self.strip_mt_data(item)
                mt_data = self.listing_all_mtdata(fold)
                self.split_mt_data(mt_data)
                # generating the mt dic
                print("Processing the mt data")
                mt_dic = self.mt_processing(mt_data)
                # plotting the shp
                print("Generating the shapefiles")
                shapefile_from_dict(input_dict=mt_dic,out_directory=out_folder,file_name="MT_data.shp")
            except:
                print("Unable to process that folder")
                continue


if __name__ == "__main__":
    input_folder = r"\\gb010587mm\GB010587MM\Glasgow 22\Glasgow_Driven_Survey\ADDR2P2\MT"
    output_folder = r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\Shapefiles\Glasgow\ADDR2P2\Regenerated"
    # output_folder = r"C:\Users\DUR94074\OneDrive - Mott MacDonald\Desktop\test"

    optionA = input("Do you want to obtain mt_shp or downsample? [1/2]")
    if optionA == str(1):
        option_ = input("Do you want to avoid plotting duplicates? [y/n] ")
        if option_.lower() == "y":
            conwy_plot = PlottingMTData(input_folder,output_folder)
            conwy_plot.plotting_shp_no_duplicates()
        else:
            conwy_plot = PlottingMTData(input_folder,output_folder)
            conwy_plot.plotting_shp()
    else:
        input_folder = r"C:\Users\DUR94074\Desktop\Glasgow"
        tst1 = reducing_shp_density(input_dir=input_folder, output_dir=input_folder,shp="merge")
        shp_data = tst1.listing_all_shpdata()
        tst1.downsample_shp(shp_data)
