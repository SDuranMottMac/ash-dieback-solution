'''
This script is used to remove false positives from the output of the pipeline.
Step 0: Identify most likely false positives.

Before using this script, please use Label Studio with the corresponding template to assess which detections are true and which are false positives.
For this, you need to upload to label studio all the images in the "detection" folder.
Once checked, export the results as Json-min. This file is used as input for this script.
The script will create 2 txt files with the false positives and the statistics of the FP and TP.
Use these to create a new field in the shapefile.
'''
# importing libraries
import os
import json
from osgeo import gdal, ogr
import shutil
import time
from pathlib import Path

def identifying_fp(input_folder):
    '''
    This function returns a list with all the fp to be removed.
    '''
    # creating a list with all the fp
    fp_list = []
    # creating a list with all the files in the folder
    files = os.listdir(input_folder)

    # looping through all the files and identify the json
    for file in files:
        if file.endswith('.json'):
            # opening the file
            with open(os.path.join(input_folder, file)) as json_file:
                # loading the json
                data = json.load(json_file)
                # looping through all the detections
                for detection in data:
                    # if the detection is not a true positive
                    if detection['choice'] == "False Positive":
                        # image name
                        image_path = detection['image']
                        image_name = os.path.basename(image_path)
                        # adding the detection to the list
                        fp_list.append(image_name[9:])
    good_fp = []
    all_files = os.listdir(os.path.join(input_folder,"Detections"))
    # returning good name
    for fp in fp_list:
        for item in os.listdir(os.path.join(input_folder,"Detections")):
            fp_split = fp.split('_')[:7]
            detection_Split = item.split('_')[:7]
            print(fp_split)
            print(detection_Split)
            if fp_split == detection_Split:
                good_fp.append(item)
                break

    # returning the fp list
    
    return(good_fp,all_files)

def removing_fp(input_folder,fp_list):
    '''
    This function removes all the false positives from the shapefile folder.
    '''
    # creating a list with the shp file in the folder
    shp_file = []
    for root, dirs, files in os.walk(input_folder):
        for fil in files:
            if fil.endswith('.shp'):
                shp_file.append(os.path.join(root,fil))

    # looping through all the files and identify the shapefile
    # opening the file
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(shp_file[0], 0)
    layer = dataSource.GetLayer()
    # looping through all the features
    total_rows = 0
    fp_found = 0
    false_positives = []
    for feature in layer:
        total_rows = total_rows + 1
        # getting the image name
        image_name = feature.GetField('Path')
        image_name = os.path.basename(image_name)
        # if the image is a fp
        
        for fp in fp_list:
            if fp[7:51] == image_name[7:51]:
                print("match!")
            # removing the feature
                fp_found = fp_found + 1
                false_positives.append(fp)
        
        
    # closing the file
    dataSource.Destroy()
    return(fp_found,total_rows,false_positives)

def metrics_txt(input_folder, fp_found, total_rows):
    '''
    This function creates a txt file with the statistics of the FP and TP.
    '''
    # creating the txt file
    with open(os.path.join(input_folder, 'metrics_FP_review.txt'), 'w') as f:
        f.write("FP: " + str(fp_found) + "\n")
        f.write("Total: " + str(total_rows - fp_found))

def more_accurate_metrics(input_folder, fp_list,files):
    global_path = Path(input_folder).parents[0]
    with open(os.path.join(global_path, 'metrics_FP_review.txt'), 'w') as f:
        f.write("FP: " + str(len(fp_list)) + "\n")
        f.write("Total Detections: " + str(len(files)))

def fp_txt(input_folder, false_positives):
    '''
    This function creates a txt file with the false positives.
    '''
    # creating the txt file
    with open(os.path.join(input_folder, 'false_positives.txt'), 'w') as f:
        f.write(", ".join(str(item) for item in false_positives))
        # for fp in false_positives:
        #     f.write(fp + "\n")

def select_likelyFP(input_folder, output_folder, threshold=0.8):
    '''
    This function selects all the images with a score lower than the threshold.
    '''
    # creating a list with all the files in the folder
    files = os.listdir(input_folder)

    # looping through all the files and identify the json
    for file_ in files:
        if file_.endswith('.png'):
            score = float(file_.split('_')[-1][:-4])/100
            if score < threshold:
                shutil.copy(os.path.join(input_folder, file_), os.path.join(output_folder,file_))

            
    
if __name__ == "__main__":
    
    # print("Instructions:")
    # print(" ")
    # time.sleep(2)
    # print("This script is used to remove false positives from the output of the pipeline")
    # print(" ")
    # time.sleep(3)
    # print("There are 2 steps to remove false positives")
    # print(" ")
    # time.sleep(3)
    # print("Step 1.0: Create a folder where you want to store images that are potential false positives")
    # print(" ")
    # print("Step 1.1: Select the folder containing all the images produced by the pipeline")
    # print(" ")
    # time.sleep(6)
    # print("Now it is time to upload these images to Label Studio under the corresponding schema and review them.")
    # print(" ")
    # time.sleep(3)
    # print("Export the results as Json-min. This file is used as input for the second part of the script.")
    # print(" ")
    # print("Place this folder in the same directory as the output of the pipeline")
    # print(" ")
    # time.sleep(6)
    # print("Step 2.0: Select the folder you just downloaded from Label Studio")
    # print(" ")
    # print("Run it!")
    # print(" ")
    # time.sleep(6)
    # print("The script will create 2 txt files with the false positives and the statistics of the FP and TP. Use these to create a new field in the shapefile")
    # time.sleep(5)


    breaking_loop = False
    count = 0
    while (breaking_loop == False):
        count = count + 1
        selecting_option = input("Select the option (reply only with the number): \n 1. Identify potential FP \n 2. Remove FP \n")
        if selecting_option == "1":
            # input folder
            input_folder = input("Input folder (containing all the images from pipeline): ")
            input_folder = r"{}".format(input_folder)
            # output folder
            output_folder = input("Output folder (to store potential FP): ")
            output_folder = r"{}".format(output_folder)
            # threshold
            score = input("Threshold: ")
            select_likelyFP(input_folder, output_folder, threshold=float(score))
            breaking_loop = True

        elif selecting_option == "2":
            # input folder
            input_folder = input("Input folder (Result folder from pipeline): ")
            input_folder = r"{}".format(input_folder)
            fp_list,files_total = identifying_fp(input_folder=input_folder)
            #fp_found, total_rows,fp_obtained = removing_fp(input_folder=input_folder,fp_list=fp_list)
            # metrics txt and fp txt
            #metrics_txt(input_folder=input_folder, fp_found=fp_found, total_rows=total_rows)
            fp_txt(input_folder=input_folder, false_positives=fp_list)
            more_accurate_metrics(input_folder=os.path.join(input_folder,"Detections"), fp_list=fp_list,files=files_total)
            print("Finished")
            breaking_loop = True

        elif count == 5:
            print("Too many attempts. Exiting...")
            break
        else:
            print("Please select a valid option: type either 1 or 2")