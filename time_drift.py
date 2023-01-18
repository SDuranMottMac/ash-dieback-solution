# importing libraries
from datetime import datetime, timedelta, timezone
import os
import pandas as pd
import pyzed.sl as sl
import time
import argparse
from tqdm import tqdm
from glob import glob

# Adding functionality

class TimeDriftEstimator():
    """
    This class aims to create a summary report with the potential time drift estimates
    """
    # Instantiating the class object
    def __init__(self,main_dir,out_dir,project):
        # instantiating variables
        self.main_dir = main_dir
        self.out_dir = out_dir
        self.project = project
    
    def obtaining_videoname_ts(self,input_vid):
        """
        Obtaining the time data from video name
        """
        # video_front_cam_14-07_14-16-58
        # obtaining time data
        if input_vid.split("_")[1] in ["front","rear"]:
            # stripping data from video name
            day = int(input_vid.split("_")[3].split("-")[0])
            month = int(input_vid.split("_")[3].split("-")[1])
            year = 2022
            hour = int(input_vid.split("_")[4].split("-")[0])
            minute = int(input_vid.split("_")[4].split("-")[1])
            second = int(input_vid.split("_")[4].split("-")[2].split(".")[0])
            #returning the data
            return(day,month,year,hour,minute,second)
        else:
            return(None,None,None,None,None,None)
    
    def obtaining_zedvid_ts(self,input_vid=None):
        """
        Obtaining the time data from the first frame of video
        """
        # instantiating an exit variable
        exit_signal = False
        # creating a camera object
        zed = sl.Camera()
        # create a InitParameters object and set configuration parameters
        input_type = sl.InputType()
        # reading svo file
        if input_vid is not None:
            input_type.set_from_svo_file(input_vid)
        # defining init parameters
        init_params = sl.InitParameters(input_t=input_type)
        init_params.camera_resolution = sl.RESOLUTION.HD2K
        init_params.camera_fps = 15
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        init_params.coordinate_units = sl.UNIT.METER
        init_params.svo_real_time_mode = False
        # Opening the camera
        err = zed.open(init_params)
        #print(err)
        if err != sl.ERROR_CODE.SUCCESS:
            err = zed.open(init_params)
            print(err)
            return(None)
        else:
            # Instantiating the runtime parameters
            runtime_parameters = sl.RuntimeParameters()

            while not exit_signal:
                if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                    # grabbing the timestamp
                    #first_image_timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE)
                    first_image_timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE)
                    first_img_ts = first_image_timestamp.get_microseconds()
                    first_img_ts =round(first_img_ts/1000000,0)
                    # breaking the loop
                    exit_signal = True
            # closing the camera
            zed.close()
            # returning the timestamp
            return(first_img_ts)
        
    def generating_timestamp(self,year,month,day,hour,minute,second):
        nano = int(0)
        dt = datetime(int(year),int(month),int(day),int(hour),int(minute),int(second),nano,tzinfo=timezone.utc)
        epoch = datetime(1970,1,1,0,0,0,0,tzinfo=timezone.utc)
        timestamp = (dt-epoch)//timedelta(microseconds=1)
        timestamp =round(timestamp/1000000,0)
        return(timestamp)

    def second_check(self,vids_list,txt):
        print('checking to see if all videos are processed')
        num_vids = len(vids_list)
        with open(txt,'r') as f:
            lines = f.readlines() 
        num_lines = len(lines)
        if num_vids > num_lines:
            print('not all videos processed - finding missing videos')
            vid = []
            for i in lines:
                i = i.split()
                vid.append(i[0])
            for item in vid:
                if item in vids_list:
                    vids_list.remove(item)
            return vids_list

        elif num_vids< num_lines:
            print('txt has more videos that videos processed. Possibly old txt.')
            return []
        else:
            print('all videos checked')
            return []
    
    def already_processed(self):
        video_name = []
        video_info = []
        #txt already exists, read txt into list to search
        with open(os.path.join(self.out_dir,str(self.project)+".txt"), "r") as textfile:
            lines = textfile.readlines()
            for line in lines:
                l = line.split()
                video_name.append(l[0])
                video_info.append(l)

        return video_name,video_info
        
        
    def obtaining_report(self):
        # defining the lists to store information
        rel_path = []
        video_name = []
        diff_ts = []
        time_drift = []
        if os.path.exists(os.path.join(self.out_dir,str(self.project)+".txt")):
            name_list, video_info = self.already_processed()
        else:
            name_list=[]

        folders = sorted(glob(self.main_dir+"/*/", recursive = True))
        route = 'R4P2'
        folders1 = [os.path.basename(os.path.dirname(i).split("\\")[-1]) for i in folders]
        start = folders1.index(route)
        glob1 = folders[start:]
        for path in glob1:
        # lopping over all the videos
            for root, dirs, files in os.walk(path, topdown=False):
                print(root)
                for name in tqdm(files):
                    if not os.path.isfile(os.path.join(self.out_dir,str(self.project)+".txt")):
                        with open(os.path.join(self.out_dir,str(self.project)+".txt"), 'w') as new_csv_file:
                            pass
                    # selecting only the svo files
                    if name.endswith(".svo"):
                        time.sleep(.1)
                        #print("Dealing with video "+str(name))
                        # appending video name
                        video_name.append(name)
                        # obtaining full path
                        video_path = os.path.join(root, name)
                        # calculating relative path
                        relative_path = os.path.relpath(video_path, self.main_dir)
                        rel_path.append(relative_path)
                        #print('checking if video has been evaluated')
                        #append info to appropriate lists if video has been processed
                        if name in name_list:
                            #print('Video already evaluated')
                            index = name_list.index(name)
                            info = video_info[index]
                            diff_ts.append(info[1])
                            time_drift.append(info[2])

                        #evaluate video if it has not been processed
                        else:
                            #skip small videos
                            size = os.path.getsize(os.path.join(root,name))/(1048576)
                            if size<30:
                                #print('Video too small')
                                diff_ts.append(0)
                                time_drift.append("TOO SMALL")
                                line_to_write = str(name)+" "+str("-")+" "+str("TOO SMALL")
                                textfile = open(os.path.join(self.out_dir,str(self.project)+".txt"), "a")
                                textfile.write(str(line_to_write) + "\n")
                                textfile.close()
                                continue
                            # obtaining video name timestamp
                            day,month,year,hour,minute,second = self.obtaining_videoname_ts(input_vid=name)
                            # turning into seconds timestamp
                            videoname_ts = int(self.generating_timestamp(year,month,day,hour,minute,second))
                        # print("The video name ts: "+str(videoname_ts))
                            # obtaining first frame ts
                            first_frame = self.obtaining_zedvid_ts(input_vid=video_path)
                            if first_frame != None:
                                first_frame = int(first_frame)
                            #print("The first frame ts: "+str(first_frame))
                            # calculating time drift
                            if first_frame == None:
                                diff_ts.append(0)
                                time_drift.append("INVALID SVO")
                                line_to_write = str(name)+" "+str("-")+" "+str("INVALID SVO")

                            else:
                                df_ts = first_frame - videoname_ts + 3600
                                diff_ts.append(df_ts)
                                if abs(df_ts) > 2:
                                    time_drift.append("TIME DRIFT")
                                    line_to_write = str(name)+" "+str(df_ts)+" "+str("TIME DRIFT")
                                else:
                                    time_drift.append("-")
                                    line_to_write = str(name)+" "+str(df_ts)+" "+str("NO")
                            textfile = open(os.path.join(self.out_dir,str(self.project)+".txt"), "a")
                            textfile.write(str(line_to_write) + "\n")
                            textfile.close()

            ''''reprocess = self.second_check(video_name,txt=os.path.join(self.out_dir,str(self.project)+".txt"))
            if len(reprocess) > 0:
                for root, dirs, files in os.walk(self.main_dir, topdown=False):
                    for name in files:
                        if name in reprocess:
                            print("Dealing with video "+str(name))
                            # appending video name
                            video_name.append(name)
                            # obtaining full path
                            video_path = os.path.join(root, name)
                            # calculating relative path
                            relative_path = os.path.relpath(video_path, self.main_dir)
                            rel_path.append(relative_path)
                            # obtaining video name timestamp
                            day,month,year,hour,minute,second = self.obtaining_videoname_ts(input_vid=name)
                            # turning into seconds timestamp
                            videoname_ts = int(self.generating_timestamp(year,month,day,hour,minute,second))
                            print("The video name ts: "+str(videoname_ts))
                            # obtaining first frame ts
                            first_frame = self.obtaining_zedvid_ts(input_vid=video_path)
                            if first_frame != None:
                                first_frame = int(first_frame)
                            print("The first frame ts: "+str(first_frame))
                            # calculating time drift
                            if first_frame == None:
                                diff_ts.append(0)
                                time_drift.append("INVALID SVO")
                                line_to_write = str(name)+" "+str("-")+" "+str("INVALID SVO")

                            else:
                                df_ts = first_frame - videoname_ts + 3600
                                diff_ts.append(df_ts)
                                if abs(df_ts) > 2:
                                    time_drift.append("TIME DRIFT")
                                    line_to_write = str(name)+" "+str(df_ts)+" "+str("TIME DRIFT")
                                else:
                                    time_drift.append("-")
                                    line_to_write = str(name)+" "+str(df_ts)+" "+str("NO")
                            textfile = open(os.path.join(self.out_dir,str(self.project)+".txt"), "a")
                            textfile.write(str(line_to_write) + "\n")
                            textfile.close()
    '''
        #read in all of txt 

        # turning all data into dataframe
        drift_report = pd.DataFrame({"Video":video_name,"Route":rel_path,"Difference (s)":diff_ts,"Time Drift":time_drift})
        drift_report.to_csv(os.path.join(self.out_dir,str(self.project)+".csv"))
        print('All Done :)')



if __name__ == "__main__":
    # Defining the parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=None, help='input path for the folder')
    parser.add_argument('--project', type=str, default=None, help='name of the csv')
    opt = parser.parse_args()

    # defining variables
    main_dir = opt.input_path
    out_dir = opt.input_path
    project = opt.project
    # Instantiating object
    reporter = TimeDriftEstimator(main_dir,out_dir,project)
    # obtaining report
    reporter.obtaining_report()

 


