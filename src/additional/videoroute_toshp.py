# This code aims to create a shapefile with the intersection of the camera and mt kit data
# importing libraries
from logging import raiseExceptions
import os
import pandas as pd
import math
import numpy as np
from datetime import datetime, timedelta, timezone
import pyzed.sl as sl
from osgeo import osr
from osgeo import ogr
import pyproj
from glob import glob

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


# Route Checker
class RouteChecker():
    """
    This class aims to plot the intersection between the cameras and the MT kit. Thus, getting to know the amount of survey collected by videos
    """

    def __init__(self, input_folder,output_folder):
        self.input_data = input_folder
        self.output_folder = output_folder
    
    def video_list(self):
        print("Creating the list with all the videos...")
        # Creating an empty list
        video_list = []
        # Populating with .svo files
        for item in os.listdir(self.input_data):
            if item.endswith(".svo"):
                video_list.append(os.path.join(self.input_data,item))
        
        return(video_list)

    def video_list_subfolders(self): 
        #returns list of all of the videos input file has folders 
        print("Creating the list with all of the videos...")
         # Creating an empty list
        video_list = []
        # Populating with .svo files
        for root, folders, files in os.walk(self.input_data, topdown=False):
            for item in files:
                if item.endswith(".svo"):
                    video_list.append(os.path.join(root,item))

        return(video_list)

    def mt_data_list(self):
        print("Creating the list with the mt data...")
        # Creating an empty list
        mt_list = []
        # Populating with .txt files
        for item in os.listdir(self.input_data):
            if item.endswith(".txt"):
                mt_list.append(os.path.join(self.input_data,item))
        
        return(mt_list)

    def mt_data_list_subfolders(self): 
        #returns list of all of the videos input file has folders 
        print("Creating the list with the mt data...")
         # Creating an empty list
        mt_list = []
        # Populating with .txt files
        for root, folders, files in os.walk(self.input_data, topdown=False):
            for item in files:
                if item.endswith(".txt"):
                    mt_list.append(os.path.join(root,item))


        return(mt_list)
    
    def listing_directories(self):

        all_folders = glob(self.input_folder+"/*/", recursive = True)
        return(all_folders)
    
    def mt_preprocessing(self, mt_list):
        print("Preprocessing the mt data")
        print("Removing info about mt kit at the top")
        # Creating an empty list 
        file_to_remove = []
        # Looping over all the mt list files
        for item in mt_list:
            # Get the file name
            val = os.path.basename(item)
            # Get the directory
            input_dir = os.path.dirname(item)
            file_to_write = []
            # Get the new file name
            name = val.split(".")[0] + "-000.txt"
            # Append the previous file to the remove list
            file_to_remove.append(item)
            # Collect the file info without the initial // sections
            with open(item, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("//"):
                        pass
                    else:
                        file_to_write.append(line)
            # Write the info into a .txt file
            with open(os.path.join(input_dir,name), 'w+') as p:
                p.writelines(file_to_write)
        # Remove original file
        for item in file_to_remove:
            os.remove(item)

        new_mt_data = self.mt_data_list()
        print("Splitting file into smaller files")
        files_to_remove = []
        for file__ in new_mt_data:
            # Get the file size
            filesize = os.path.getsize(file__)
            # Get the file name
            filename = os.path.basename(file__)
            # Get the input directory
            input_dir = os.path.dirname(file__)
            # If size exceed 500 MB
            if filesize > 500000000:# 500MB
                #split it
                with open(file__, 'r') as f:
                    lines = f.readlines()
                    header = lines[0]
                    amnt = int(filesize/500000000)
                    split_files = np.array_split(lines[1:], amnt)

                    for idx, val in enumerate(split_files):
                        with open(os.path.join(input_dir,(filename.split(".")[0]+"00{}.txt".format(idx))), 'w+') as p:
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
                df_raw = pd.read_csv(item)
                # Iterate over the data in the mt kit file        
                for index,row in df_raw.iterrows():
                    # we generate the timestamp
                    timestamp = generating_timestamp(nano=row['UTC_Nano'],year=row['UTC_Year'],month=row['UTC_Month'],day=row['UTC_Day'],hour=row['UTC_Hour'],minute=row['UTC_Minute'],second=row['UTC_Second'])
                    # add the timestamp as key to the dict
                    if timestamp in mit_data:
                        if ((math.isnan(mit_data[timestamp]["latitude"])) and (not math.isnan(row['Latitude']))):
                            mit_data[timestamp] = {"latitude":row['Latitude'],"longitude":row['Longitude'],"Altitude":row['Altitude'],"Pitch":row['Pitch'],"Roll":row['Roll']}
                            #mit_data[timestamp] = {"latitude":row['Latitude'],"longitude":row['Longitude'],"Altitude":row['Altitude']}
                        else:  
                            continue
                    else:
                        mit_data[timestamp] = {"latitude":row['Latitude'],"longitude":row['Longitude'],"Altitude":row['Altitude'],"Pitch":row['Pitch'],"Roll":row['Roll']}
                        #mit_data[timestamp] = {"latitude":row['Latitude'],"longitude":row['Longitude'],"Altitude":row['Altitude']}
            except:
                try:
                    # read the file
                    df_raw = pd.read_csv(item,sep=",")
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
                except:
                    raise Exception("I am not able to read the file, most likely due to an incorrect separator")

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
    
    def obtaining_coordinate_frames(self, video_list, mit_data):
        print("Obtaining the coordinates")
        
        # Creating a dictionary with all the data
        polyline_dict = dict()
        # Creating an ID for the frames
        id_frame = 0
        # Looping over all the videos
        for video_ in video_list:
            # Specify SVO Init Parameters
            init_params = sl.InitParameters()
            init_params.set_from_svo_file(str(video_))
            init_params.svo_real_time_mode = False  # Don't convert in realtime
            init_params.coordinate_units = sl.UNIT.METER  # Use milliliter units (for depth measurements)
            init_params.camera_resolution = sl.RESOLUTION.HD2K  # Use HD1080 video mode
            init_params.camera_fps = 15
            # Specify SVO Runtime Parameters
            rt_param = sl.RuntimeParameters()
            rt_param.sensing_mode = sl.SENSING_MODE.STANDARD

            # Create ZED objects
            zed = sl.Camera() 
            # Open the SVO file specified as a parameter
            err = zed.open(init_params)
            if err != sl.ERROR_CODE.SUCCESS:
                print("Error while opening the camera")
                zed.close()
                continue
            nb_frames = zed.get_svo_number_of_frames()
            i = 0
            # Extracting the data from the video
            while i < nb_frames:
                i = i + 1
                if zed.grab(rt_param) == sl.ERROR_CODE.SUCCESS:
                    timestamp_cam = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE) 
                    ts_nanoseconds = timestamp_cam.get_nanoseconds()
                    value = round(float(ts_nanoseconds/1000000000),2)
                    print("timestamp is: ",str(value))
                    k = 0
                    while True:
                        if value in mit_data:
                            lat = mit_data[value]["latitude"]
                            long = mit_data[value]["longitude"]
                            break
                        else:
                            value = round(value - 0.01,2)
                            k += 1
                            if k%1000000 == 0:
                                print("May be stuck in loop. Think about aborting.",k)


                            
                            
                    polyline_dict[id_frame]= {"latitude":lat,"longitude":long}
                    id_frame = id_frame + 1
            
        
            # Close the camera
            zed.close()
        return(polyline_dict)
    
    def shapefile_creator(self,polyline_dict):
        print("Generating the shapefile")
        # first, we need to define a few variables
        shapepath = os.path.join(self.output_folder,"Route_covered.shp")

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
        for key in polyline_dict:
            id_ = key
            latitude = polyline_dict[key]["latitude"]
            longitude = polyline_dict[key]["longitude"]

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
    input_folder = r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\Surveyors_SD_Card_Data\Conwy\Conwy_part_2\Route 1P4ADD"
    output_folder = r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\Shapefiles\Conway\Route 1P4ADD"
    
    Route_conwy = RouteChecker(input_folder=input_folder,output_folder=output_folder)
    video_list = Route_conwy.video_list_subfolders() ##
    mt_data_list = Route_conwy.mt_data_list_subfolders()
    Route_conwy.mt_preprocessing(mt_data_list)
    mt_data_list = Route_conwy.mt_data_list_subfolders()
    mti_data = Route_conwy.mt_processing(mt_data_list)
    polyline_dict = Route_conwy.obtaining_coordinate_frames(video_list, mti_data)
    Route_conwy.shapefile_creator(polyline_dict)












