'''
This code aims to crea a shapefile with the intersection of the camera and mt kit data. This code takes an input folder, output folder, and log txt*
Input folder is a folder of route folders, output folder is a folder where route shapefiles can be created in their respective route folders. Log txt currently does not work
 '''

# importing libraries
from logging import raiseExceptions
from msilib.schema import Class
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
import logging as logger
import time


def mt_data_collection(path_list):
    mt_list = []
    for p in path_list:
        for root, folders, files in os.walk(p,topdown=True):
            for item in files:
                if item.endswith(".txt"):
                    mt_list.append(os.path.join(root,item))
                    
        return mt_list

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
    return(interpolated_dict)


class CreateSHP():

    def __init__(self, input_folder,output_folder,txt_file):
        self.input_folder = input_folder
        self.output_folder = output_folder 
        self.txt_file = txt_file

    def video_list_subfolders(self,route_path): 
        #returns list of all of the videos input file has folders 
        logger.info("Creating the list with all of the videos...")
         # Creating an empty list
        video_list = []
        # Populating with .svo files
        for root, folders, files in os.walk(route_path):
            for item in files:
                if item.endswith(".svo"):
                    video_list.append(os.path.join(root,item))

        return(video_list)

    def mt_data_collection(self,path_list):
        #function takes a list of paths and finds all txt files in each path and adds txt path to list. List of all txt files returned
        mt_list = []
        for p in path_list:
            for root, folders, files in os.walk(p,topdown=True):
                for item in files:
                    if item.endswith(".txt"):
                        mt_list.append(os.path.join(root,item))
        
                    
        return mt_list

    def mt_data_collect(self,path):
        mt_path_list = []
        for root, folders, files in os.walk(path):
                for item in files:
                    if item.endswith(".txt"):
                        j = os.path.join(root,item)
                        mt_path_list.append(j)
        return mt_path_list

    def mt_data_list_subfolders(self,route_path,in_folders): 
        #returns list of all of the mt data in the current folder and the two folders around  
        logger.info("Creating the list with the mt data from the current folder and surrounding folders...")
         # Creating an empty list
        f_index = in_folders.index(route_path)

        if len(in_folders)== 1:
            #added in for the test 1/9/22
            mt_list = self.mt_data_collect(route_path)
            return(mt_list)

        elif f_index == 0:
            #if folder is the first folder, only pull mt txt files from next folder
            next_fold = in_folders[f_index+1]
            #first half
            next_fold_paths = self.mt_data_collect(next_fold)
            next_fold_paths = next_fold_paths[:len(next_fold_paths)//2]
            fold = self.mt_data_collect(route_path)
            mt_list = fold+next_fold_paths
            return(mt_list)

        elif f_index == len(in_folders)-1:
            #if folder is the last folder, only pull mt txt files from previous folder
            prev_fold = in_folders[f_index-1]
            #last half
            prev_fold_paths = self.mt_data_collect(prev_fold)
            prev_fold_paths = prev_fold_paths[len(prev_fold_paths)//2:]
            fold = self.mt_data_collect(route_path)
            mt_list = prev_fold_paths +fold
            
            return(mt_list)

        else:
            #else, grab mt txt files from the previous and the next folder
            prev_fold = in_folders[f_index-1]
            next_fold = in_folders[f_index+1]
            #last half
            prev_fold_paths = self.mt_data_collect(prev_fold)
            prev_fold_paths = prev_fold_paths[len(prev_fold_paths)//2:]
            #first half
            next_fold_paths = self.mt_data_collect(next_fold)
            next_fold_paths = next_fold_paths[:len(next_fold_paths)//2]

            fold = self.mt_data_collect(route_path)
            mt_list = prev_fold_paths+fold+next_fold_paths
            return(mt_list)


    def mt_preprocessing(self, mt_list,route_path,in_folders):
        logger.info("Preprocessing the mt data")
        logger.info("Removing info about mt kit at the top")
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

        new_mt_data = self.mt_data_list_subfolders(route_path,in_folders)
        logger.info("Splitting file into smaller files")
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
        logger.info("Post-processing the mt data")
        # create a dictionary to store all data
        mit_data = dict()
        # loop over all the items in the list
        print("creating mit_data dictionary")
        for item in mt_list:
            print(item)
            try:
                # read the file
                print("reading file")
                df_raw = pd.read_csv(item)
                # Iterate over the data in the mt kit file
                print("generating timestamp")   
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
                        
            except:
                try:
                    print("reading file in exception")
                    # read the file
                    df_raw = pd.read_csv(item,sep=",")
                    # Iterate over the data in the mt kit file
                    print("generating timestamp in exception")        
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
                except:
                    raise Exception("I am not able to read the file, most likely due to an incorrect separator")
        print("creating timestamp keys")
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
        print("going into interpolating_df")
        mit_data= interpolating_df(input_dict=mit_data)
        return(mit_data)

    def obtaining_coordinate_frames(self, video_list, mit_data):
        logger.info("Obtaining the coordinates")
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
                logger.info("Video skipped. Error while opening the camera of video,%s ", str(video_))
                zed.close()
                continue
            nb_frames = zed.get_svo_number_of_frames()
            i = 0
            num_skips = 0 
            # Extracting the data from the video
            while i < nb_frames:
                i +=1
                if zed.grab(rt_param) == sl.ERROR_CODE.SUCCESS:
                    timestamp_cam = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE) 
                    ts_nanoseconds = timestamp_cam.get_nanoseconds()
                    value = round(float(ts_nanoseconds/1000000000),2)
                    if i%1000 == 1:
                        print("timestamp is: ",str(value))
                    k = 0
                    skip = False
                    while True:
                        if value in mit_data:
                            lat = mit_data[value]["latitude"]
                            long = mit_data[value]["longitude"]
                            break
                        else:
                            value = round(value - 0.01,2)
                            k += 1
                            if k%1000000 == 0:
                                print("If stuck in the loop, frame will be skipped",k/1000000)
                            if k > 15000000:
                                print("skip to next frame")
                                logger.info("Skipping to next frame")
                                skip = True
                                num_skips += 1 
                                break

                    if skip == False:
                         polyline_dict[id_frame]= {"latitude":lat,"longitude":long}
                         id_frame = id_frame + 1
                    
                    if num_skips > 10:
                        logger.info(" skips occured. Good chance video is not included in MT data. Video skipped is %s", str(video_))
                        break
            
            # Close the camera
            zed.close()
        print("done with polyline_dict")
        return(polyline_dict)

    def shapefile_creator(self,polyline_dict,route_path_out):
        logger.info("Generating the shapefile")
        # first, we need to define a few variables
        shapepath = os.path.join(route_path_out,"Route_covered.shp")

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


    def make_shp(self):
        print("Collecting directory names")
        in_folders = sorted(glob(self.input_folder+"/*/", recursive = True))
        out_folders = sorted(glob(self.output_folder+"/*/", recursive = True))
        txt_file = self.txt_file
        in_folder_names = [] #list of names of routes i.e Route 1P2
        out_folder_names = []

        print("Creating names of folders")
        for f in in_folders:
            in_folder_names.append(os.path.basename(os.path.dirname(f).split("\\")[-1]))
            
        for f in out_folders:
            out_folder_names.append(os.path.basename(os.path.dirname(f).split("\\")[-1]))

        #Check for matching outfolder, create folder if one does not exist
        print("Creating missing out folders")
        for name in in_folder_names:
            if name not in out_folder_names:
                out_folder_names.append(name)
                path = os.path.join(self.output_folder,name)
                out_folders.append(path)
                os.mkdir(path)

        print("Processing each folder")
        for route_path in in_folders: #full path being sent through loop
            #check to see if it has been processed
            #check to see if file has been processed by another machine
            print("checking txt file")
            #skip file if file is in txt file
            print(route_path)
            ftxt = open(txt_file,'r')
            lines = ftxt.readlines()
            modified_lines = [os.path.dirname(elem).split("\\")[-1] for elem in lines]
            modified_route = os.path.dirname(route_path).split("\\")[-1]

            if modified_route in modified_lines:
                    print("file already processed")
                    ftxt.close()
                    continue
            else:
                ftxt.close()

            with open(txt_file,'a') as ftxt:
                ftxt.write("\n")
                ftxt.write(route_path)

            print("Checking to see if shp file has been made")
            # file_name = os.path.basename(route_path)

            # #if the file name is empty ie path ends in \, remove \ then find filename
            # if len(file_name) == 0: 
            #     route_path1 = route_path[:-1]
            #     file_name = os.path.basename(route_path1)
            #     print(file_name)

            route_path_out = os.path.join(self.output_folder,modified_route)
            print("route out is ",route_path_out)
            if os.path.isfile(os.path.join(route_path_out,"Route_covered.shp")):
                print("Route_covered.shp is already made. Moving to next route.")
                continue

            #initiate shapefile creation
            current_route = str(os.path.basename(route_path))
            print("Processing ", current_route)
            logger.info("Currently handling %s", current_route)
            try:
                #Return list of videos
                print("creating video list")
                video_list = self.video_list_subfolders(route_path=route_path)
                #list of mt_data from current path and surrounding folders. 
                print("creating mt data lists")
                mt_data_list = self.mt_data_list_subfolders(route_path=route_path,in_folders=in_folders)
                print(mt_data_list)
                 
                time.sleep(10)
                print("preprocessing mt data")
                self.mt_preprocessing(mt_data_list,route_path, in_folders)
                print("recreating mt data list")
                mt_data_list = self.mt_data_list_subfolders(route_path=route_path,in_folders=in_folders)
                print("post processing mt data")
                mti_data = self.mt_processing(mt_data_list)
                print("creating polyline_dict")
                try:
                    
                    polyline_dict = self.obtaining_coordinate_frames(video_list,mti_data)
                
                except:
                    print("unable to process folder at polyline_dict")
                    logger.info("Unable to process folder") 
                    continue

                print("generating shapefile")
                self.shapefile_creator(polyline_dict,route_path_out)

            except:
                print("unable to process folder")
                logger.info("Unable to process folder") 
                continue


if __name__ == "__main__":
    input_folder = r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\Surveyors_SD_Card_Data\Ceredigion\Ceredigion 22"
    output_folder = r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\Shapefiles\Ceredigion"
    logfiletxt = r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\Shapefiles\automation_log.txt"
    txt_file = r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\Shapefiles\t_log.txt"

    logger.basicConfig(filename = logfiletxt,filemode = 'a',format='%(message)s,%(lineno)d')
    Route_creater = CreateSHP(input_folder=input_folder,output_folder=output_folder,txt_file=txt_file)
    Route_creater.make_shp()

