# importing libraries
from codecs import latin_1_decode
import os
from math import radians, cos, sin, asin, sqrt
from pickletools import long1
from timeit import timeit
import numpy as np
import timeit
from tqdm import tqdm
import pathlib
from datetime import datetime, timedelta, timezone
import pandas as pd
import math

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

def generating_timestamp(nano,year,month,day,hour,minute,second):
    # turn this column into integers
    nano = int(nano/1000)
    # turn the time values into a datetime object
    dt = datetime(int(year),int(month),int(day),int(hour),int(minute),int(second),nano,tzinfo=timezone.utc)
    # datetime object 0
    epoch = datetime(1970,1,1,0,0,0,0,tzinfo=timezone.utc)
    # obtaining the timestampt from current to 0
    timestamp = (dt-epoch)//timedelta(microseconds=1)
    # rounding the timestamp
    timestamp =round(timestamp/1000000,2)
    # returning ts
    return(timestamp)

def getList(dictionary):
    return dictionary.keys()

def interpolating_df(input_dict):
    # first we need to transform the dictionary into a dataframe
    df = pd.DataFrame.from_dict(input_dict,orient="index")
    # now we are going to get the interpolated index
    idxs = list(df.index.values)
    starting = float(min(idxs))
    ending = float(max(idxs))
    assert (ending > starting)
    # creating a new index
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

    # turning the interpolated dataframe to a dictionary
    interpolated_dict = interpolated_merged.to_dict("index")
    # returning the dictioanry
    return(interpolated_dict)


class DuplicateID():
    """
    This class creates an object that identifies all the duplicates videos per project
    """

    def __init__(self, input_folder,output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
    
    def listing_mtfiles(self):
         # Creating an empty list
        mt_files = []
        # Populating the list with all the svi videos
        for path, subdirs, files in os.walk(self.input_folder):
            for name in files:
                if name.endswith(".txt") and (len(name.split("_")) == 0 or len(name.split("_")) == 1):
                    if os.path.join(path, name) not in mt_files:
                        mt_files.append(os.path.join(path, name))
                elif name == "Tracked.txt":
                    continue
                elif (name.endswith(".txt")) and (name.split("_")[1] != "monitoring"):
                    if os.path.join(path, name) not in mt_files:
                        mt_files.append(os.path.join(path, name))
        return(mt_files)
    
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
    
    def processing_mtData(self, mt_file):
        # lists to store data
        mit_data = dict()
        # Reading the mt data file
        df_raw = pd.read_csv(mt_file)
        # Iterating over the dataframe and collecting information
        for index,row in df_raw.iterrows():
            # we generate the timestamp
            timestamp = generating_timestamp(nano=row['UTC_Nano'],year=row['UTC_Year'],month=row['UTC_Month'],day=row['UTC_Day'],hour=row['UTC_Hour'],minute=row['UTC_Minute'],second=row['UTC_Second'])
            # add the timestamp as key to the dict
            if timestamp in mit_data:
                if ((math.isnan(mit_data[timestamp]["latitude"])) and (not math.isnan(row['Latitude']))):
                    mit_data[timestamp] = {"latitude":row['Latitude'],"longitude":row['Longitude'],"Altitude":row['Altitude']}
                else:  
                    continue
            else:
                mit_data[timestamp] = {"latitude":row['Latitude'],"longitude":row['Longitude'],"Altitude":row['Altitude']}
        # get a list with all the keys   
        all_keys = getList(mit_data)
        del_keys = []
        # remove some keys with nan data
        for key in all_keys:
            if math.isnan(mit_data[key]["latitude"]):
                del_keys.append(key)
        for key in del_keys:
            del mit_data[key]
        # interpolating the mt data so all gaps are covered
        mit_data= interpolating_df(input_dict=mit_data)
        # returning a dictionary with the mt data processed
        return(mit_data)
    
    def obtaining_duplicatesTS(self):
        print("Instantiating the arrays")
        # Instantiating the np arrays to store the data
        ts_array = np.array([])
        longitude_array = np.array([])
        latitude_array = np.array([])
        # Obtaining a list of the mt files
        print("Listing the mt files")
        mt_files = self.listing_mtfiles()
        # preprocessing large files
        print("Preprocessing large files")
        self.split_mt_data(mt_files)
        # Updating the list with the new files
        mt_files = self.listing_mtfiles()
        # Instantiating the list of duplicates
        duplicateTS = []
        print("Looping over all mt files")
        for item in mt_files:
            print("Dealing with file "+str(item))
            # preprocessing the item - stripping potential badly formatted characters
            self.strip_mt_data(item)
            # processing the info in the file
            mt_info = self.processing_mtData(item)
            # assessing duplicates
            duplicate_grabbed = False
            for ix,key in enumerate(mt_info):
                # checking whether it is the first time we append info
                if ((ix == 0) or (ix == 50)):
                    ts_array = np.append(ts_array,np.array(key))
                    longitude_array = np.append(longitude_array,np.array(mt_info[key]["longitude"]))
                    latitude_array = np.append(latitude_array,np.array(mt_info[key]["latitude"]))
                elif (ix % 50 == 0):
                    # perform the calculations
                    lon1 = longitude_array
                    lat1 = latitude_array
                    lon2 = np.array(mt_info[key]["longitude"])
                    lat2 = np.array(mt_info[key]["latitude"])
                    sep = haversine_np(lon1,lat1,lon2,lat2)
                    eval_matrix = np.any(sep < 30)
                    # evaluating whether any previous point is less than 15 meters
                    if eval_matrix:
                        # obtaining the index
                        pos_points = np.where(sep < 30)
                        # assessing whether these occurred at least 30 seconds earlier (or more)
                        pos_ts = np.take(ts_array,pos_points[0])
                        # if any happened more than 30 seconds earlier
                        if np.any(np.absolute(pos_ts-key) > 60):
                            print("duplicate identified: "+str(key))
                            print("Corresponding with: "+str(pos_ts))
                            # duplicate identified
                            if duplicate_grabbed:
                                # select and append all the keys from the previous up to now
                                last_key = int(duplicateTS[-1]*100)
                                current_key = int(key*100)
                                ts_ = list(range(last_key,current_key))
                                duplicateTS=duplicateTS + [float(round(elem/100,2)) for elem in ts_]
                            else:
                                # First duplicate found
                                duplicateTS.append(key)
                                duplicate_grabbed = True
                    else:
                        # Turn off the duplicate grabbed variable
                        duplicate_grabbed = False
                        # As it is not a duplicate, we append info into the np arrays
                        ts_array = np.append(ts_array,np.array(key))
                        longitude_array = np.append(longitude_array,np.array(mt_info[key]["longitude"]))
                        latitude_array = np.append(latitude_array,np.array(mt_info[key]["latitude"]))
            
            # we disable the interpolation between files
            duplicate_grabbed = False

        # saving the duplicate list into a txt file
        textfile = open(os.path.join(self.output_folder,"duplicate_list.txt"), "w")
        for element in duplicateTS:
            textfile.write(str(element) + "\n")
        textfile.close()
        # returning the list
        return(duplicateTS)

    def obtaining_duplicatesTS_Lat_Long(self):
        print("Instantiating the arrays")
        # Instantiating the np arrays to store the data
        ts_array = np.array([])
        longitude_array = np.array([])
        latitude_array = np.array([])
        # Obtaining a list of the mt files
        print("Listing the mt files")
        mt_files = self.listing_mtfiles()
        # preprocessing large files
        print("Preprocessing large files")
        self.split_mt_data(mt_files)
        # Updating the list with the new files
        mt_files = self.listing_mtfiles()
        # Instantiating the list of duplicates
        duplicateTS = []
        lat_dup = []
        long_dup = []
        print("Looping over all mt files")
        for item in mt_files:
            print("Dealing with file "+str(item))
            # preprocessing the item - stripping potential badly formatted characters
            self.strip_mt_data(item)
            # processing the info in the file
            mt_info = self.processing_mtData(item)
            # assessing duplicates
            duplicate_grabbed = False
            for ix,key in enumerate(mt_info):
                # checking whether it is the first time we append info
                if ((ix == 0) or (ix == 50)):
                    ts_array = np.append(ts_array,np.array(key))
                    longitude_array = np.append(longitude_array,np.array(mt_info[key]["longitude"]))
                    latitude_array = np.append(latitude_array,np.array(mt_info[key]["latitude"]))
                elif (ix % 50 == 0):
                    # perform the calculations
                    lon1 = longitude_array
                    lat1 = latitude_array
                    lon2 = np.array(mt_info[key]["longitude"])
                    lat2 = np.array(mt_info[key]["latitude"])
                    sep = haversine_np(lon1,lat1,lon2,lat2)
                    eval_matrix = np.any(sep < 30)
                    # evaluating whether any previous point is less than 15 meters
                    if eval_matrix:
                        # obtaining the index
                        pos_points = np.where(sep < 30)
                        # assessing whether these occurred at least 30 seconds earlier (or more)
                        pos_ts = np.take(ts_array,pos_points[0])
                        # if any happened more than 30 seconds earlier
                        if np.any(np.absolute(pos_ts-key) > 60):
                            print("duplicate identified: "+str(key))
                            print("Corresponding with: "+str(pos_ts))
                            lat_dup.append(mt_info[key]["latitude"])  
                            long_dup.append(mt_info[key]["longitude"])
                            # duplicate identified
                            if duplicate_grabbed:
                                # select and append all the keys from the previous up to now
                                last_key = int(duplicateTS[-1]*100)
                                current_key = int(key*100)
                                ts_ = list(range(last_key,current_key))
                                duplicateTS=duplicateTS + [float(round(elem/100,2)) for elem in ts_]
                            else:
                                # First duplicate found
                                duplicateTS.append(key)
                                duplicate_grabbed = True
                    else:
                        # Turn off the duplicate grabbed variable
                        duplicate_grabbed = False
                        # As it is not a duplicate, we append info into the np arrays
                        ts_array = np.append(ts_array,np.array(key))
                        longitude_array = np.append(longitude_array,np.array(mt_info[key]["longitude"]))
                        latitude_array = np.append(latitude_array,np.array(mt_info[key]["latitude"]))
            
            # we disable the interpolation between files
            duplicate_grabbed = False

        # saving the duplicate list into a txt file
        textfile = open(os.path.join(self.output_folder,"duplicate_list.txt"), "w")
        for element in duplicateTS:
            textfile.write(str(element) + "\n")
        textfile.close()
        # returning the list
        return(duplicateTS,lat_dup,long_dup)
    
    def generate_shp(self,lat,long):
        from osgeo import osr
        from osgeo import ogr
        import pyproj
        # first, we need to define a few variables
        shapepath = os.path.join(self.output_folder,"duplicates.shp")

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
        for id_ in range(len(lat)):
            id_ = id_
            latitude = lat[id_]
            longitude = long[id_]
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
                


if __name__ == "__main__":

    # Defining directories
    input_folder = r"\\gb010587mm\Software_dev\Ash_Dieback_Solution\Tests\duplicate_test"
    output_folder = r"\\gb010587mm\Software_dev\Ash_Dieback_Solution\Tests\duplicate_test"
    # instantiating the object
    duplicate_checker = DuplicateID(input_folder,output_folder)
    dup,lat,long = duplicate_checker.obtaining_duplicatesTS_Lat_Long()
    duplicate_checker.generate_shp(lat=lat,long=long)





