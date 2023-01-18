# importing libraries
from fileinput import filename
import os
import pyzed.sl as sl
import datetime
from PIL import ImageStat, Image
import time
import numpy as np
import random
import pandas as pd
import cv2
from statistics import mode
import math
from csv import writer


def brightness(im_file):
 
    #img = Image.fromarray(np.uint8(im_file))
    img = Image.fromarray(im_file)
    stat = ImageStat.Stat(img)
    r,g,b = stat.mean

    return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))


# creating functionality
class FileChecker():
    """
    This class aims to check the incoming files and generate a report for further inspection
    """
    def __init__(self, input_folder,output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
    
    def list_all_videos(self):
        """
        List all the videos in the input folder with the .svo extension
        """
        print("Listing all the videos")
         # Creating an empty list
        vids = []
        # Populating the list with all the svi videos
        for path, subdirs, files in os.walk(self.input_folder):
            for name in files:
                if name.endswith(".svo"):
                    if os.path.join(path, name) not in vids:
                        vids.append(os.path.join(path, name))
        return(vids)
    def calculating_route(self,path):
        """
        This method returns the route number from the absolute path
        """
        print("Calculating the route")
        
        all_directories = path.split("\\")
        route_dir = None
        for dir in all_directories:
            if dir[:4].lower() == "rout":
                route_dir = dir
        
        if route_dir == None:
            for dir in all_directories:
                if dir[:3].lower() == "rou":
                    route_dir = dir
        print("Route: "+str(route_dir))
        return(route_dir)
    
    def extracting_cam_orientation(self,path):
        file_name = os.path.basename(path)
        print("file name: "+str(file_name))
        orientation = file_name.split("_")[1]
        return(orientation)
    
    def obtain_filesize(self, path):
        size_bytes = os.path.getsize(path)
        return(float(size_bytes/1000000))

    def estimating_video_length(self, path,orientation):
        # Specify SVO path parameter
        init_params = sl.InitParameters()
        init_params.set_from_svo_file(str(path))
        init_params.svo_real_time_mode = False  # Don't convert in realtime
        init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)

        # Create ZED objects
        zed = sl.Camera()
        # Open the SVO file specified as a parameter
        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(err)
            nb_frames = 0
            lng = str(datetime.timedelta(seconds=int(nb_frames/15)))
            zed.close()

        else:
            # Calculate number of frames
            nb_frames = zed.get_svo_number_of_frames()
            lng = str(datetime.timedelta(seconds=int(nb_frames/15)))
            zed.close()

        return(lng)
    
    def check_length(self, length, orientation):
        hours,minutes,seconds = length.split(":")
        if int(minutes) == 8 and (orientation == "front"):
            return("Y")
        elif (int(minutes) == 7) and (int(seconds) > 55) and (orientation == "front"):
            return("Y")
        elif int(minutes) == 4 and (orientation == "rear"):
            return("Y")
        elif (int(minutes) == 3) and (int(seconds) > 55) and (orientation == "rear"):
            return("Y")
        else:
            return("N")
    
    def brightness_check(self,path):
        print("Starting brightness calculation")
        skip_to_next = False
         # Specify SVO path parameter
        init_params = sl.InitParameters()
        init_params.set_from_svo_file(str(path))
        init_params.svo_real_time_mode = False  # Don't convert in realtime
        init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)

        # Create ZED objects
        zed = sl.Camera()

        # Open the SVO file specified as a parameter
        print("Trying to open the camera")
        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(err)
            zed.close()
            skip_to_next = True
        if skip_to_next:
            print("Invalid video")
            return("INVALID VIDEO","INVALID VIDEO", "INVALID VIDEO","INVALID VIDEO")
        else:

            left_image = sl.Mat()
            right_image = sl.Mat()
            rt_param = sl.RuntimeParameters()
            rt_param.sensing_mode = sl.SENSING_MODE.STANDARD
            diff_brithness = []
            brighter_cam = []
            darkness = []
            diff_HSV = []
            n_loops = 0
            open_loop = True
            while open_loop:
                print("Loop number ", str(n_loops))
                n_loops = n_loops + 1
                try:
                    print("Grabbing new image")
                    new_frame = zed.grab(rt_param)
                    time.sleep(0.05)
                    print("Grabbed")
                except Exception as e:
                    print(e)
                    continue

                if new_frame == sl.ERROR_CODE.SUCCESS:
                    print("Image grabbed "+str(n_loops))
                    time.sleep(0.01)

                    if random.random() < 0.03:
                        # Retrieve SVO images
                        print("Retrieving left")
                        zed.retrieve_image(left_image, sl.VIEW.LEFT)
                        print("Retrieving right") 
                        zed.retrieve_image(right_image, sl.VIEW.RIGHT)
                        width_2K = 2208
                        height_2K = 1242
                        width_sbs_2K = width_2K
                        # Prepare side by side image container equivalent to CV_8UC4
                        svo_image_sbs_rgba_left = np.zeros((height_2K, width_sbs_2K, 4), dtype=np.uint8)
                        svo_image_sbs_rgba_right = np.zeros((height_2K, width_sbs_2K, 4), dtype=np.uint8)
                        print("Getting left data to array")
                        svo_image_sbs_rgba_left[:, :, :] = left_image.get_data()
                        print("Getting right data to array")
                        svo_image_sbs_rgba_right[:, :, :] = right_image.get_data()
                        # Convert SVO image from RGBA to RGB
                        print("cvt color RGB left image")
                        ocv_image_sbs_rgb_left = cv2.cvtColor(svo_image_sbs_rgba_left, cv2.COLOR_RGBA2RGB)
                        print("RGB color right image")
                        ocv_image_sbs_rgb_right = cv2.cvtColor(svo_image_sbs_rgba_right, cv2.COLOR_RGBA2RGB)
                        print("Going for the random estimation")
                        print("Calculating left brightness")
                        left_brightness = brightness(ocv_image_sbs_rgb_left)
                        if left_brightness < 30:
                            darkness.append(1)
                        print("Calculating right brightness")
                        right_brightness = brightness(ocv_image_sbs_rgb_right)
                        print("Calculating difference in brightness")
                        diff_brithness.append(abs(left_brightness - right_brightness))
                        # EVALUATE THE HSV DIFFERENCE
                        green1 = ocv_image_sbs_rgb_left[:, :, 1:2]
                        green2 = ocv_image_sbs_rgb_right[:, :, 1:2]
                        blue1 = ocv_image_sbs_rgb_left[:, :, :1]
                        blue2 = ocv_image_sbs_rgb_right[:, :, :1]
                        red1 = ocv_image_sbs_rgb_left[:, :, 2:]
                        red2 = ocv_image_sbs_rgb_right[:, :, 2:]

                        # computing the mean
                        g1_mean = np.mean(green1)
                        g2_mean = np.mean(green2)
                        b1_mean = np.mean(blue1)
                        b2_mean = np.mean(blue2)
                        r1_mean = np.mean(red1)
                        r2_mean = np.mean(red2)

                        if ((abs(float(g1_mean) - float(g2_mean))+abs(float(b1_mean) - float(b2_mean))+abs(float(r1_mean) - float(r2_mean)))/3) > 50:
                            diff_HSV.append(1)
                        
                        try:
                            if left_brightness > right_brightness:
                                print("the left camera is brighter")
                                brighter_cam.append("Left")
                            else:
                                print("the right camera is brighter")
                                brighter_cam.append("Right")
                        except Exception as e:
                                print(e)
                                print("error while trying to estimate which camera is brighter")
                                brighter_cam.append("Unknown")
                        
                        
                    else:
                        print("Not evaluated")
                    
                    if len(diff_brithness) == 9:
                        print("Already got 9 brightness estimation, closing camera")
                        print(diff_brithness)
                        print(brighter_cam)
                        # Close the camera
                        try:
                            print("Attepmting to close camera after obtaining the 3 estimations")
                            zed.close()
                            open_loop = False
                            print("Leaving the loop")
                            
                        except Exception as e:
                            print(e)
                            print("not able to close the camera")
                            open_loop = False
                            
                else:
                    print("Error while grabbing a new image")
                    n_loops = n_loops + 1
                    if n_loops > 1000:
                        print("Already got to 1000 loops so quitting...")
                        try:
                            print("Attepmting to close camera")
                            zed.close()
                            open_loop = False
                        except Exception as e:
                            print(e)
                            print("Error while closing the camera")
                            open_loop = False

                    continue

                if n_loops > 1000:
                    print("Already got to 1000 loops so quitting...")
                    try:
                        print("Attepmting to close camera")
                        zed.close()
                        open_loop = False
                    except Exception as e:
                        print(e)
                        print("Error while closing the camera")
                        open_loop = False
            try:
                if (float(np.average(np.array(diff_brithness))) > 10) and len(darkness)>0 and len(diff_HSV)>0:
                    print("Difference in brightness")
                    return("Difference in brightness",mode(brighter_cam),"Dark frame detected","Different HSV detected")
                elif (float(np.average(np.array(diff_brithness))) > 10) and len(darkness)>0 and len(diff_HSV)==0:
                    print("Difference in brightness")
                    return("Difference in brightness",mode(brighter_cam),"Dark frame detected","Same HSV detected")
                elif (float(np.average(np.array(diff_brithness))) > 10) and len(darkness)==0 and len(diff_HSV)>0:
                    print("Difference in brightness")
                    return("Difference in brightness",mode(brighter_cam),"No Dark frames detected", "Different HSV detected")
                elif (float(np.average(np.array(diff_brithness))) > 10) and len(darkness)==0 and len(diff_HSV)==0:
                    print("Difference in brightness")
                    return("Difference in brightness",mode(brighter_cam),"No Dark frames detected", "Same HSV detected")
                elif (float(np.average(np.array(diff_brithness))) <= 10) and len(darkness)==0 and len(diff_HSV)>0:
                    print("similar brightness")
                    return("Similar brightness","Same","No Dark frames detected", "Different HSV detected")
                elif (float(np.average(np.array(diff_brithness))) <= 10) and len(darkness)>0 and len(diff_HSV)>0:
                    print("similar brightness")
                    return("Similar brightness","Same","Dark frame detected", "Different HSV detected")
                elif (float(np.average(np.array(diff_brithness))) <= 10) and len(darkness)>0 and len(diff_HSV)==0:
                    print("similar brightness")
                    return("Similar brightness","Same","Dark frame detected", "Same HSV detected")
                else:
                    print("similar brightness")
                    return("Similar brightness","Same","No Dark frames detected","Same HSV detected")
            except:
                print("error while estimating whether the brightness is similar")
                return("ERROR","Unknown","Unknown","Unknown")
    
    def append_list_as_row(self,file_name, list_of_elem):
        # Open file in append mode
        with open(file_name, 'a+', newline='') as write_obj:
            # Create a writer object from csv module
            csv_writer = writer(write_obj)
            # Add contents of list as last row in the csv file
            csv_writer.writerow(list_of_elem)
    
    def generating_spreadsheet(self):
        route=[]
        camera_orientation = []
        zed_name = []
        file_size = []
        length_video = []
        check_length_vid = []
        check_brightness = []
        brighter_cam = []
        dark_frames = []
        diff_HSV = []

        vid_id = 0
        list_v = self.list_all_videos()

        # checking whether a csv file already exists
        if not os.path.isfile(os.path.join(self.output_folder,"ZED_check.csv")):
            headers = ["Route", "Camera Orientation", "Zed File Name","File Size [Mb]","Length of the video [hh:mm:ss]","Is the video approx. 8 minutes or longer? (Y/N)","Bright camera","Dark frames","Diff HSV"]
            # if the file does not exists, we create an empty one
            with open(os.path.join(self.output_folder,"ZED_check.csv"), 'w',newline='') as my_new_csv_file:
                csvwriter = writer(my_new_csv_file)
                csvwriter.writerow(headers)
                my_new_csv_file.close()
        # checking if a list of already processed video exists
        if os.path.isfile(os.path.join(self.output_folder,"processed_videos.txt")):
            # opening the file in read mode
            processed_file = open(os.path.join(self.output_folder,"processed_videos.txt"), "r")
            # reading the file
            data_processed = processed_file.read()
            # replacing end of line('/n') with ' ' and
            # splitting the text it further when '.' is seen.
            data_processed_videos = data_processed.replace('\n', ';').split(";")
            # printing the data
            print(data_processed_videos)
            processed_file.close()
        else:
            data_processed_videos = []
            with open(os.path.join(self.output_folder,"processed_videos.txt"), 'w') as new_csv_file:
                pass

        for vid in list_v:
            # if the video has already been processed
            if vid in data_processed_videos:
                continue
            data_processed_videos.append(vid)
            print("Dealing with video "+str(vid_id))
            # ID of the video
            vid_id = vid_id + 1
            # Route of the video
            route.append(self.calculating_route(path=vid))
            # camera orientation
            camera_orientation.append(self.extracting_cam_orientation(path=vid))
            # File name
            zed_name.append(os.path.basename(vid))
            # File size
            file_size.append(self.obtain_filesize(path=vid))
            try:
                length_video.append(self.estimating_video_length(path=vid,orientation=camera_orientation[-1]))              
                # check length length, orientation
                check_length_vid.append(self.check_length(length = length_video[-1],orientation = camera_orientation[-1]))
            except:
                length_video.append("Not manipulable")
                check_length_vid.append("Not manipulable")
                check_brightness.append("Not manipulable")
                brighter_cam.append("Not manipulable")
                dark_frames.append("Not manipulable")
                diff_HSV.append("Not manipulable")
                continue
            try:
                diff_brightnes,brigh_cm,drk_img,dif_hsv = self.brightness_check(path=vid)
                check_brightness.append(diff_brightnes)
                brighter_cam.append(brigh_cm)
                dark_frames.append(drk_img)
                diff_HSV.append(dif_hsv)
            except:
                check_brightness.append("INVALID VIDEO")
                check_length_vid[-1] = "INVALID VIDEO"
                brighter_cam.append("INVALID VIDEO")
                dark_frames.append("INVALID VIDEO")
                diff_HSV.append("INVALID VIDEO")
                continue

            # info to append
            to_append = [route[-1],camera_orientation[-1],zed_name[-1],file_size[-1],length_video[-1],check_length_vid[-1],check_brightness[-1],brighter_cam[-1],dark_frames[-1],diff_HSV[-1]]
            self.append_list_as_row(os.path.join(self.output_folder,"ZED_check.csv"),to_append)
            # append video to duplicate 
            textfile = open(os.path.join(self.output_folder,"processed_videos.txt"), "a")
            textfile.write(str(vid) + "\n")
            textfile.close()
        

if __name__ == "__main__":
    # Defining folders
    input_folder = [r"\\gb010587mm\New_Seagate\GB010587MM\Ceredigion 22",r"\\gb010587mm\New_Seagate\GB010587MM\Conwy 22",r"\\gb010587mm\New_Seagate\GB010587MM\North Yorkshire 22"]
    output_folder = [r"\\gb010587mm\New_Seagate\GB010587MM\Ceredigion 22",r"\\gb010587mm\New_Seagate\GB010587MM\Conwy 22",r"\\gb010587mm\New_Seagate\GB010587MM\North Yorkshire 22"]

    for idx,item in enumerate(input_folder):
        spread1 = FileChecker(input_folder[idx],output_folder[idx])
        for i in range(5):
            print("Round "+str(i))
            try:
                spread1.generating_spreadsheet()
            except:
                print("Error in video, continue with new loop!")
                continue


