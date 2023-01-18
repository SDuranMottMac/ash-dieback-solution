# importing libraries
# ===================
from datetime import datetime, timedelta, timezone
import math
from math import radians, cos, sin, asin, sqrt
import os
import pandas as pd
from pyproj import Proj
import matplotlib.pyplot as plt
import numpy as np
import statistics
import glob
import time
from tqdm import tqdm


# functionality
# =============
def generating_timestamp(nano,year,month,day,hour,minute,second):
    nano = int(nano/1000)
    dt = datetime(int(year),int(month),int(day),int(hour),int(minute),int(second),nano,tzinfo=timezone.utc)
    epoch = datetime(1970,1,1,0,0,0,0,tzinfo=timezone.utc)
    timestamp = (dt-epoch)//timedelta(microseconds=1)
    timestamp =round(timestamp/1000000,2)
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


def collecting_mtidata(input_directory):
    # lists to store data
    mit_data = dict()

    for subdir, dirs, files in os.walk(input_directory):
        for item in files:
            if item.endswith('.txt'):
                if (item.split("_")[1]=="monitoring"):
                    continue
                else:
                    try:
                     df_raw = pd.read_csv(os.path.join(subdir,item))
                    except:
                        try:
                            df_raw = pd.read_csv(os.path.join(subdir,item), delimiter = " ", on_bad_lines="skip", header=None)
                            df_raw = df_raw.drop([1,2,3],axis=1)
                            df_raw = df_raw.drop([0,1,2,3,4,5],axis=0)
                            df_raw = df_raw[0].str.split(',', expand=True)
                            df_raw = df_raw.reset_index(drop=True)
                        except:
                            print("MT kit data badly formatted")
                            break
                for index,row in df_raw.iterrows():
                        # we generate the timestamp
                        
                        timestamp = generating_timestamp(nano=row['UTC_Nano'],year=row['UTC_Year'],month=row['UTC_Month'],day=row['UTC_Day'],hour=row['UTC_Hour'],minute=row['UTC_Minute'],second=row['UTC_Second'])
                        # add the timestamp as key to the dict
                        if timestamp in mit_data:
                            continue
                        else:
                            mit_data[timestamp] = {"latitude":row['Latitude'],"longitude":row['Longitude'],"Altitude":row['Altitude']}
    
    all_keys = getList(mit_data)
    del_keys = []
    for key in all_keys:
        if math.isnan(mit_data[key]["latitude"]):
            del_keys.append(key)
    for key in del_keys:
        del mit_data[key]

    mit_data= interpolating_df(input_dict=mit_data)
    return(mit_data)

def object_angle(image_width, detection_x,HFOV=110):
    ''' 
    This function estimates the angle that the detection forms with the camera

    - image_width -> width of the image in pixels
    - detection_x -> coordinate X of the center of the detection in pixels
    - HFOV -> Horizontal field of view of the camera
    '''
    H_angle = ((int(detection_x) - int(image_width/2))/(int(image_width/2)))*(int(HFOV/2))
    
    return(H_angle) # Degrees

def coordinate_calc_bearing(current_lon,current_lat,previous_lon,previous_lat,distance,angle):
    ''' 
    This function uses the bearing angle and the distance to work out the coordinates of the detection.
    This function assumes that the camera are facing parallel to the motion of the car

    '''
    #The first step is converting the coordinates into UTM
    p = Proj(proj='utm',zone=30,ellps='WGS84', preserve_units=False)
    lon_0, lat_0 = p(previous_lon, previous_lat)
    lon_1, lat_1 = p(current_lon, current_lat)
    
    
    # The horizontal angle of the detection needs to be turned into radian 
    angle_rad = angle * math.pi/180

    #now we need to calculate the bearing
    d_lon0 = lon_1-lon_0
    d_lat0 = lat_1 - lat_0
    if d_lat0 == 0:
        d_lat0 = 0.0000001
    else:
        d_lat0 = d_lat0
    
    if (d_lon0 > 0) & (d_lat0 > 0):
        b0_angle = math.atan(d_lon0/d_lat0)
        b_angle = b0_angle + angle_rad  # that's the bearing in radians
       
    
    elif (d_lon0 < 0) & (d_lat0 > 0):
        b0_angle = 2*math.pi - math.atan(abs(d_lon0/d_lat0))
        b_angle = b0_angle + angle_rad 

    elif (d_lon0 > 0) & (d_lat0 < 0):
        b0_angle = math.pi - math.atan(abs(d_lon0/d_lat0))
        b_angle = b0_angle + angle_rad 
    
    else:
        b0_angle = math.pi + math.atan(abs(d_lon0/d_lat0))
        b_angle = b0_angle + angle_rad

    d_lon1 = distance * sin(b_angle)
    d_lat1 = distance * cos(b_angle)

    lon_2 = lon_1 + d_lon1
    lat_2 = lat_1 + d_lat1
    
    #And now we need to take them back to decimal degrees
    lon2, lat2 = p(lon_2, lat_2, inverse=True)

    return(lon2, lat2)

def ts_to_coordinates(ts,mit_data,distance,img_with,center_x):
    '''
    This function aims to obtain the coordinates of the detection based on the coordinates of mtkit and distance
    '''
    # checking whether the timestamp is in the mt data dictionary
    if ts in mit_data:
        # grabbing the current latitude longitude
        current_lat = mit_data[ts]["latitude"]
        current_long = mit_data[ts]["longitude"]
        # searching for the previous latitude longitude
        value = round(ts - 0.01,2)
        ts_count = 0
        while True:
            # counting how many times it searches for it
            ts_count = ts_count + 1
            # checking whether it has looped for over 1000 times (10 seconds)
            if ts_count == 1000:
                # if so, stop it and return none
                previous_lat = None
                previous_long = None
                break
            # if value found, grab it
            elif value in mit_data:
                previous_lat = mit_data[value]["latitude"]
                previous_long = mit_data[value]["longitude"]
                break
            # else, go for the previous value
            else:
                value = round(value - 0.01,2)
        # if value not found, return none
        if (previous_lat == None) or (previous_long == None):
            return(None, None)
        # else, proceed with the calculation
        else:
            # we calculate the object angle
            objectangle = object_angle(image_width=img_with, detection_x=center_x,HFOV=110)
            # calculate the detections
            detection_latitude, detection_longitude = coordinate_calc_bearing(current_lon=current_long,current_lat=current_lat,previous_lon=previous_long,previous_lat=previous_lat,distance=distance,angle=objectangle)
            # return the values
            return(detection_latitude, detection_longitude)
    
    else:
        
        value = round(ts - 0.01,2)
        ts_count = 0
        while True:
            ts_count = ts_count + 1
            if ts_count == 1000:
                current_lat = None
                current_long = None
                break
            elif value in mit_data:
                current_lat = mit_data[value]["latitude"]
                current_long = mit_data[value]["longitude"]
                break
            else:
                value = round(value - 0.01,2)
                
        prev_value = round(value - 0.01,2)
        ts_count = 0
        while True:
            ts_count = ts_count + 1
            if ts_count == 1000:
                previous_lat = None
                previous_long = None
                break
            elif prev_value in mit_data:
                previous_lat = mit_data[value]["latitude"]
                previous_long = mit_data[value]["longitude"]
                break
            else:
                prev_value = round(prev_value - 0.01,2)
        if (previous_lat == None) or (previous_long == None):
            return(None, None)  
        elif (current_lat == None) or (current_long == None):
            return(None, None) 
        else:         
            # we calculate the object angle
            objectangle = object_angle(image_width=img_with, detection_x=center_x,HFOV=110)
            # calculate the detections
            detection_latitude, detection_longitude = coordinate_calc_bearing(current_lon=current_long,current_lat=current_lat,previous_lon=previous_long,previous_lat=previous_lat,distance=distance,angle=objectangle)
            # return the values
            return(detection_latitude, detection_longitude)

def ts_to_rear_coordinates(ts,mit_data,distance,img_with,center_x):
    '''
    This function aims to obtain the coordinates of the detection based on the coordinates of mtkit and distance
    '''

    if ts in mit_data:
        
        current_lat = mit_data[ts]["latitude"]
        current_long = mit_data[ts]["longitude"]
        value = round(ts - 0.01,2)
        ts_count = 0
        while True:
            ts_count = ts_count + 1
            if ts_count == 1000:
                previous_lat = None
                previous_long = None
                break
            elif value in mit_data:
                previous_lat = mit_data[value]["latitude"]
                previous_long = mit_data[value]["longitude"]
                break
            else:
                value = round(value - 0.01,2)
        if (previous_lat == None) or (previous_long == None):
            return(None, None)
        else:
            # we calculate the object angle
            objectangle = object_angle(image_width=img_with, detection_x=center_x,HFOV=110)
            # transform to rear angle - degrees
            rear_objectangle = objectangle + 180
            if rear_objectangle > 360:
                rear_objectangle = rear_objectangle - 360
            # projecting the distance
            # calculate the detections
            projected_distance = distance*cos(0.349066)
            # calculate the detections
            detection_latitude, detection_longitude = coordinate_calc_bearing(current_lon=current_long,current_lat=current_lat,previous_lon=previous_long,previous_lat=previous_lat,distance=projected_distance,angle=rear_objectangle)
            # return the values
            return(detection_latitude, detection_longitude)
    
    else:
        
        value = round(ts - 0.01,2)
        ts_count = 0
        while True:
            ts_count = ts_count + 1
            if (ts_count == 1000):
                current_lat = None
                current_long = None
                break
            elif value in mit_data:
                current_lat = mit_data[value]["latitude"]
                current_long = mit_data[value]["longitude"]
                break
            else:
                value = round(value - 0.01,2)
                
        prev_value = round(value - 0.01,2)
        ts_count = 0
        while True:
            ts_count = ts_count + 1
            if (ts_count == 1000):
                previous_lat = None
                previous_long = None
                break
            elif prev_value in mit_data:
                previous_lat = mit_data[value]["latitude"]
                previous_long = mit_data[value]["longitude"]
                break
            else:
                prev_value = round(prev_value - 0.01,2)
        if (previous_lat == None) or (previous_long == None):
            return(None, None)  
        elif (current_lat == None) or (current_long == None):
            return(None, None) 
        else:      
            # we calculate the object angle
            objectangle = object_angle(image_width=img_with, detection_x=center_x,HFOV=110)
            # transform to rear angle - degrees
            rear_objectangle = objectangle + 180
            if rear_objectangle > 360:
                rear_objectangle = rear_objectangle - 360
            # projecting the distance
            projected_distance = distance*cos(0.349066)
            # calculate the detections
            detection_latitude, detection_longitude = coordinate_calc_bearing(current_lon=current_long,current_lat=current_lat,previous_lon=previous_long,previous_lat=previous_lat,distance=projected_distance,angle=rear_objectangle)
            # return the values
            return(detection_latitude, detection_longitude)


def ts_to_projected_coordinates(ts,mit_data,distance,img_with,img_height,center_x,center_y,VFOV=70):
    '''
    This function aims to obtain the coordinates of the detection based on the coordinates of mtkit and distance
    '''
    # calculating the tilted angle
    tilted_a = ((float(VFOV)/2)*abs((float(img_height)/2)-float(center_y)))/(float(img_height)/2)
    # obtaining the car coordinates
    if ts in mit_data:
        
        current_lat = mit_data[ts]["latitude"]
        current_long = mit_data[ts]["longitude"]
        value = round(ts - 0.01,2)
        while True:
            if value in mit_data:
                previous_lat = mit_data[value]["latitude"]
                previous_long = mit_data[value]["longitude"]
                break
            else:
                value = round(value - 0.01,2)
               
        # we calculate the bearing angle
        objectangle = object_angle(image_width=img_with, detection_x=center_x,HFOV=110)
        # calculate the detections
        projected_distance = distance*cos(tilted_a)
        detection_latitude, detection_longitude = coordinate_calc_bearing(current_lon=current_long,current_lat=current_lat,previous_lon=previous_long,previous_lat=previous_lat,distance=projected_distance,angle=objectangle)
        # return the values
        return(detection_latitude, detection_longitude)
    
    else:
        
        value = round(ts - 0.01,2)
        while True:
            if value in mit_data:
                current_lat = mit_data[value]["latitude"]
                current_long = mit_data[value]["longitude"]
                break
            else:
                value = round(value - 0.01,2)
                
        prev_value = round(value - 0.01,2)
        while True:
            if prev_value in mit_data:
                previous_lat = mit_data[value]["latitude"]
                previous_long = mit_data[value]["longitude"]
                break
            else:
                prev_value = round(prev_value - 0.01,2)
                
        # we calculate the object angle
        objectangle = object_angle(image_width=img_with, detection_x=center_x,HFOV=110)
        # calculate the detections
        projected_distance = distance*cos(tilted_a)
        detection_latitude, detection_longitude = coordinate_calc_bearing(current_lon=current_long,current_lat=current_lat,previous_lon=previous_long,previous_lat=previous_lat,distance=projected_distance,angle=objectangle)
        # return the values
        return(detection_latitude, detection_longitude)



def collecting_mtidata_pitch(input_directory):
    # lists to store data
    mit_data = dict()

    for subdir, dirs, files in os.walk(input_directory):
        for item in files:
            # we get all the mt kit files
            if item.endswith('.txt'):
                if (item.split("_")[1]=="monitoring"):
                    continue
                else:
                    try:
                        df_raw = pd.read_csv(os.path.join(subdir,item))
                    except:
                        try:
                            df_raw = pd.read_csv(os.path.join(subdir,item), delimiter = " ", on_bad_lines="skip", header=None)
                            df_raw = df_raw.drop([1,2,3],axis=1)
                            df_raw = df_raw.drop([0,1,2,3,4,5],axis=0)
                            df_raw = df_raw[0].str.split(',', expand=True)
                            df_raw = df_raw.reset_index(drop=True)
                        except:
                            print("MT kit data badly formatted")
                            break
                    for index,row in df_raw.iterrows():
                        # we generate the timestamp
                        timestamp = generating_timestamp(nano=row['UTC_Nano'],year=row['UTC_Year'],month=row['UTC_Month'],day=row['UTC_Day'],hour=row['UTC_Hour'],minute=row['UTC_Minute'],second=row['UTC_Second'])
                        # add the timestamp as key to the dict
                        if timestamp in mit_data:
                            continue
                        else:
                            mit_data[timestamp] = {"latitude":row['Latitude'],"longitude":row['Longitude'],"Altitude":row['Altitude'],"Pitch":row['Pitch'],"Roll":row['Roll']}
                            
    all_keys = getList(mit_data)
    del_keys = []
    print(mit_data)
    for key in all_keys:
        if math.isnan(mit_data[key]["latitude"]):
            del_keys.append(key)
    for key in del_keys:
        del mit_data[key]
    
    mit_data= interpolating_df(input_dict=mit_data)
    return(mit_data)


def reduce_mt_files(input_directory):
    # lists to store data
    mit_data = dict()

    for subdir, dirs, files in os.walk(input_directory):
        for item in files:
            if item.endswith('.txt'):
                if os.path.getsize(os.path.join(input_directory, item)) > 500000:
                    #do something
                    pass
                else:
                    #leave it
                    pass
def strip_mt_data(input_directory):
    file_to_remove = []
    print("Assessing whether MT files are properly formatted")
    for idx, val in enumerate(tqdm(os.listdir(input_directory))):
        if val.endswith('.txt'):
            file_to_write = []
            name = val.split(".")[0] + "-000.txt"
            file_to_remove.append(os.path.join(input_directory,val))
            with open(os.path.join(input_directory,val), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("//"):
                        pass
                    else:
                        file_to_write.append(line)

            with open(os.path.join(input_directory,name), 'w+') as p:
                p.writelines(file_to_write)
    
    for item in file_to_remove:
        os.remove(item)
        

def split_mt_data(input_dir):
    print("Assessing whether the size of the files are adequate")
    files_to_remove = []
    for file__ in tqdm(os.listdir(input_dir)):
        filepath = os.path.join(input_dir, file__)
        filesize = os.path.getsize(filepath)
        if filesize > 500000000 and file__.endswith(".txt"):# 500MB
            #split it
            with open(os.path.join(filepath), 'r') as f:
                lines = f.readlines()
                header = lines[0]
                amnt = int(filesize/500000000)
                split_files = np.array_split(lines[1:], amnt)

                for idx, val in enumerate(split_files):
                    with open(os.path.join(input_dir,(file__.split(".")[0]+"00{}.txt".format(idx))), 'w+') as p:
                        p.write(header)
                        p.writelines(val)
            
            # append it to the list
            files_to_remove.append(os.path.join(input_dir,file__))
    
    # removing files too large
    for item in files_to_remove:
        os.remove(item)


def collecting_mtidata_pitch_lessfreq(input_directory):
    # lists to store data
    mit_data = dict()

    for subdir, dirs, files in os.walk(input_directory):
        for item in files:
            # we get all the mt kit files
            if item.endswith('.txt'):
                
                if len(item.split("_")) < 2:
                    try:
                        df_raw = pd.read_csv(os.path.join(subdir,item))
                    except:
                        try:
                            df_raw = pd.read_csv(os.path.join(subdir,item), delimiter = " ", on_bad_lines="skip", header=None)
                            df_raw = df_raw.drop([1,2,3],axis=1)
                            df_raw = df_raw.drop([0,1,2,3,4,5],axis=0)
                            df_raw = df_raw[0].str.split(',', expand=True)
                            df_raw = df_raw.reset_index(drop=True)
                        except:
                            print("MT kit data badly formatted")
                            print(item)
                            break
                    for index,row in df_raw.iterrows():
                        # we generate the timestamp
                        timestamp = generating_timestamp(nano=row['UTC_Nano'],year=row['UTC_Year'],month=row['UTC_Month'],day=row['UTC_Day'],hour=row['UTC_Hour'],minute=row['UTC_Minute'],second=row['UTC_Second'])
                        # add the timestamp as key to the dict
                        if timestamp in mit_data:
                            if ((math.isnan(mit_data[timestamp]["latitude"])) and (not math.isnan(row['Latitude']))):
                                mit_data[timestamp] = {"latitude":row['Latitude'],"longitude":row['Longitude'],"Altitude":row['Altitude'],"Pitch":row['Pitch'],"Roll":row['Roll']}
                            else:  
                                continue
                        else:
                            mit_data[timestamp] = {"latitude":row['Latitude'],"longitude":row['Longitude'],"Altitude":row['Altitude'],"Pitch":row['Pitch'],"Roll":row['Roll']}
                            
                elif (item.split("_")[1]=="monitoring"):
                    continue
                else:
                    try:
                        df_raw = pd.read_csv(os.path.join(subdir,item))
                    except:
                        try:
                            df_raw = pd.read_csv(os.path.join(subdir,item), delimiter = " ", on_bad_lines="skip", header=None)
                            df_raw = df_raw.drop([1,2,3],axis=1)
                            df_raw = df_raw.drop([0,1,2,3,4,5],axis=0)
                            df_raw = df_raw[0].str.split(',', expand=True)
                            df_raw = df_raw.reset_index(drop=True)
                        except:
                            print("MT kit data badly formatted")
                            print(item)
                            break
                    for index,row in df_raw.iterrows():
                        # we generate the timestamp
                        timestamp = generating_timestamp(nano=row['UTC_Nano'],year=row['UTC_Year'],month=row['UTC_Month'],day=row['UTC_Day'],hour=row['UTC_Hour'],minute=row['UTC_Minute'],second=row['UTC_Second'])
                        # add the timestamp as key to the dict
                        if timestamp in mit_data:
                            if ((math.isnan(mit_data[timestamp]["latitude"])) and (not math.isnan(row['Latitude']))):
                                mit_data[timestamp] = {"latitude":row['Latitude'],"longitude":row['Longitude'],"Altitude":row['Altitude'],"Pitch":row['Pitch'],"Roll":row['Roll']}
                            else:  
                                continue
                        else:
                            mit_data[timestamp] = {"latitude":row['Latitude'],"longitude":row['Longitude'],"Altitude":row['Altitude'],"Pitch":row['Pitch'],"Roll":row['Roll']}
                            
    all_keys = getList(mit_data)
    del_keys = []

    for key in all_keys:
        if math.isnan(mit_data[key]["latitude"]):
            del_keys.append(key)
    for key in del_keys:
        del mit_data[key]
    
    mit_data= interpolating_df(input_dict=mit_data)
    return(mit_data)

def last_modified_file(folder_input):
    files = os.listdir(folder_input)
    paths = [os.path.join(folder_input, basename) for basename in files]
    last_file = max(paths, key=os.path.getctime)
    ti_c = os.path.getctime(last_file)
    return(ti_c)


def check_loop(ti_c,elapsed=300):
    time_now = time.time()
    if ((time_now - ti_c)> 300):
        return(True)
    else:
        return(False)

def estimating_abs_height(distance, elev_car, height_pix, height_real, bottom_y,pitch_angle,height_image):
    '''
    distance - distance from the car to the tree
    elev_car - elevation of the car
    height_pix - height of the tree in pixels
    height_real - height of the tree in meters
    bottom_y - y coordinate of the bottom of the bounding box
    pitch_angle - pitch angle of the accelerometer
    height_image - height of the image in pixels
    '''
    if (pitch_angle > 0):
        if (bottom_y < height_image/2):

            d = distance*sin(abs(pitch_angle))
            gap = (height_image/2-bottom_y)*height_real/height_pix

            abs_elev = elev_car+d+height_real+gap
            return(int(abs_elev))
        
        else:
            d = distance*sin(abs(pitch_angle))
            gap = (bottom_y-height_image/2)*height_real/height_pix
            abs_elev = elev_car+d+height_real-gap
            return(int(abs_elev))
    else:
        if (bottom_y > height_image/2):
            d = distance*sin(abs(pitch_angle))
            gap = (bottom_y-height_image/2)*height_real/height_pix
            abs_elev = elev_car-d+height_real-gap
            return(int(abs_elev))
        else:
            d = distance*sin(abs(pitch_angle))
            gap = (height_image/2-bottom_y)*height_real/height_pix
            abs_elev = elev_car-d+height_real+gap
            return(int(abs_elev))

if __name__ == '__main__':       

    #mtidata = collecting_mtidata(input_directory=r'\\gb010587mm\Software_dev\Ash_Dieback_Solution\with_mtkit')
    import json
    mti_data = collecting_mtidata(input_directory=r'\\gb010587mm\Software_dev\Ash_Dieback_Solution\with_mtkit')
    with open('data.json', 'w') as fp:
        json.dump(mti_data, fp,  indent=4)
                        

