# importing libraries
import os
import pandas as pd
import geopandas as gpd
import geopy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import matplotlib.pyplot as plt
import plotly_express as px

# adding functionality
def get_video(image_name):
    ''' this function aims to extract the video name from the image name'''
    if image_name.lower().startswith("left"):
        if (len(image_name.split("_")) == 1):
            videoname = image_name.split("-")[1]
        else:
            videoname = image_name.split("_")[0][4:]
        
        return(videoname)
    
    elif image_name.lower().startswith("right"):
        if (len(image_name.split("_")) == 1):
            videoname = image_name.split("-")[1]
        else:
            videoname = image_name.split("_")[0][4:]
        
        return(videoname)

    else:
        videoname = image_name.split("_")[0]
        return(videoname)

def get_coordinates(image_name):
    ''' This function gets coordinates from image name'''
    if image_name.lower().startswith("left"):
        if (len(image_name.split("_")) == 1):
            coord = (image_name.split("(")[1].split(",")[0],image_name.split("(")[1].split(",")[1])
        else:
            coord = (image_name.split("_")[2],image_name.split("_")[3].split(".jpg")[0])
        
        return(coord)
    
    elif image_name.lower().startswith("right"):
        if (len(image_name.split("_")) == 1):
            coord = (image_name.split("(")[1].split(",")[0],image_name.split("(")[1].split(",")[1])
        else:
            coord = (image_name.split("_")[2],image_name.split("_")[3].split(".jpg")[0])
        
        return(coord)

    else:
        coord = (image_name.split("(")[1].split(",")[0],image_name.split("(")[1].split(",")[1])
        return(coord)
    
def get_geoinfo(coord):
    '''from latitude longitude get additional positional info'''
    locator = Nominatim(user_agent="myGeocoder")
    coordinates = str(coord[0])+", "+str(coord[1])
    #print(coordinates)
    location = locator.reverse(coordinates)
    geoinfo = location.raw['address']
    
    # we get the info we need
    try:
        city = geoinfo['town']
    except:
        try:
            city = geoinfo['village']
        except:
            city = "Unknown"
    try:
        road = geoinfo['road']
    except:
        road = "Unknown"
    try:
        county = geoinfo['county']
    except:
        county= "Unknown"
    try:
        country = geoinfo['country']
    except:
        country = "Unknown"
    try:
        postcode = geoinfo['postcode']
    except:
        postcode = "Unknown"

    return(city,road,county,country,postcode)

def get_all_images(directory1):
    images = []
    for path, subdirs, files in os.walk(directory1):
        for name in files:
            if name.endswith(".jpg"):
                images.append(name)
    
    return(images)

def opentxt_append(txt,image,img_list):

    if os.path.isfile(txt):
        images_lists = img_list
        # with open(txt) as file_:
        #     for line in file_:
        #         images_lists.append(line.rstrip()[0])
        if image in images_lists:
            return("next")
        else:
            return("append")
    else:
        f = open(txt, "x")
        f.close()
        return("write")

def list_images_txt(txt):
    images_lists = []
    if os.path.isfile(txt):
        
        with open(txt) as file_:
            for line in file_:
                images_lists.append(line.split(";")[0])
        print(images_lists)
        return(images_lists)
    else:
        print(images_lists)
        return(images_lists)
    
    

def workflow(imagedir,outfolder):

    images = get_all_images(directory1=imagedir)
    df_name = "data_info.txt"
    file_path = os.path.join(outfolder,df_name)
    img_lst = list_images_txt(file_path)
    for img in images:
        # check what we need to do
        response = opentxt_append(file_path,img,img_lst)
        if (response == "next"):
            continue
        elif (response == "append"):
            # get video
            videoname = get_video(image_name=img)
            # get coordinates
            voord = get_coordinates(image_name=img)
            # get geoinfo
            city,road,county,country,postcode = get_geoinfo(coord=voord)
    
            to_append = str(img)+";"+str(videoname)+";"+str(city)+";"+str(road)+";"+str(county)+";"+str(country)+";"+str(postcode)
            #print(to_append)
            with open(file_path, 'a',encoding="utf-8") as fd:
                fd.write(f'\n{to_append}')

        else:
            videoname = get_video(image_name=img)
            # get coordinates
            voord = get_coordinates(image_name=img)
            # get geoinfo
            city,road,county,country,postcode = get_geoinfo(coord=voord)
    
            to_append = str(img)+";"+str(videoname)+";"+str(city)+";"+str(road)+";"+str(county)+";"+str(country)+";"+str(postcode)
            with open(file_path, 'w') as fd:
                fd.write(f'{to_append}')
    
    print("All Images Copied!!")

 

if __name__ == "__main__":
    # the code navigates through all the images in the directory and subdirectory, and creates the outfile file in the out directory
    # containing info about the location and image/video names
    image_directory = r'\\gb010587mm\ash_dev\data\azure_backup\dataset_train\yolov5\V2\images'
    output_dir = r'\\gb010587mm\ash_dev\data\azure_backup\dataset_train\yolov5\V2'
    workflow(imagedir=image_directory,outfolder=output_dir)





    













