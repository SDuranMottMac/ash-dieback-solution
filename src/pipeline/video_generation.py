'''
This file contains a series of functions to deal with the video generation
'''
import sys
import pyzed.sl as sl
import numpy as np
import cv2
import os

import time
def progress_bar(percent_done, bar_length=50):
    done_length = int(bar_length * percent_done / 100)
    bar = '=' * done_length + '-' * (bar_length - done_length)
    sys.stdout.write('[%s] %f%s\r' % (bar, percent_done, '%'))
    sys.stdout.flush()



def turning_video_oneimage(input_vid,output_video,vid_resolution):
    
    svo_input_path = input_vid
    output_path = output_video

    # Specify SVO path parameter
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(svo_input_path))
    init_params.svo_real_time_mode = False  # Don't convert in realtime
    init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)

    # Create ZED objects
    zed = sl.Camera()

    # Open the SVO file specified as a parameter
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        sys.stdout.write(repr(err))
        zed.close()
        exit()

    
    # Get image size
    image_size = zed.get_camera_information().camera_resolution
    camera_infos = zed.get_camera_information()
    width_HD = 1920
    height_HD = 1080
    width_sbs_HD = width_HD

    # Prepare side by side image container equivalent to CV_8UC4
    svo_image_sbs_rgba_HD = np.zeros((height_HD, width_sbs_HD, 4), dtype=np.uint8)
    
    
    # Get image size
    image_size = zed.get_camera_information().camera_resolution
    width_2K = 2208
    height_2K = 1242
    width_sbs_2K = width_2K

    # Prepare side by side image container equivalent to CV_8UC4
    svo_image_sbs_rgba_2K = np.zeros((height_2K, width_sbs_2K, 4), dtype=np.uint8)

    # Prepare single image containers
    left_image = sl.Mat()
   
    if vid_resolution == "HD":
        video_writer = cv2.VideoWriter(str(output_path),cv2.VideoWriter_fourcc('M', '4', 'S', '2'),zed.get_camera_information().camera_fps,(width_sbs_HD, height_HD))
    else:
        video_writer = cv2.VideoWriter(str(output_path),cv2.VideoWriter_fourcc('M', '4', 'S', '2'),zed.get_camera_information().camera_fps,(width_sbs_2K, height_2K))

    if not video_writer.isOpened():
        sys.stdout.write("OpenCV video writer cannot be opened. Please check the .avi file path and write permissions.\n")
        zed.close()
        exit()

    rt_param = sl.RuntimeParameters()
    rt_param.sensing_mode = sl.SENSING_MODE.STANDARD

    # Start SVO conversion to AVI/SEQUENCE
    sys.stdout.write("Converting SVO... Use Ctrl-C to interrupt conversion.\n")

    nb_frames = zed.get_svo_number_of_frames()

    n_loops = 0

    while True:
        
        if zed.grab(rt_param) == sl.ERROR_CODE.SUCCESS:
            svo_position = zed.get_svo_position()

            # Retrieve SVO images
            zed.retrieve_image(left_image, sl.VIEW.LEFT) 
            
            # Copy the left image to the left side of SBS image
            
            if vid_resolution == "HD":
                svo_image_sbs_rgba_HD[:, :, :] = left_image.get_data()
                # Convert SVO image from RGBA to RGB
                ocv_image_sbs_rgb = cv2.cvtColor(svo_image_sbs_rgba_HD, cv2.COLOR_RGBA2RGB)
            
           
            else:
                svo_image_sbs_rgba_2K[:, :, :] = left_image.get_data()
                # Convert SVO image from RGBA to RGB
                ocv_image_sbs_rgb = cv2.cvtColor(svo_image_sbs_rgba_2K, cv2.COLOR_RGBA2RGB)

             # Write the RGB image in the video
            video_writer.write(ocv_image_sbs_rgb)

            # Display progress
            progress_bar((svo_position + 1) / nb_frames * 100, 30)

            # Check if we have reached the end of the video
            if svo_position >= (nb_frames - 1):  # End of SVO
                sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
                break
    
        n_loops = n_loops + 1

        if ((n_loops - (svo_position)) > 100):  # End of SVO
            sys.stdout.write("\nLoop was exited but video generated\n")
            video_writer.release()
            
            break
        
    
    video_writer.release()

    zed.close()

def decimate_frame(vid_path, frame_number,out_dir,file_name):
    '''
    This function saves the frame indicated from a given video
    - vid_path = absolute path of the video
    - file_name = the image that is going to be found within the video and saved
    - out_dir = directory where frames are saved
    ''' 
    try:
        frame_number = frame_number
        vidcap = cv2.VideoCapture(vid_path)
        vidcap.set(cv2.CAP_PROP_POS_FRAMES,frame_number)
        _, image = vidcap.read()
        out_file = os.path.join(out_dir,file_name)
        try:
            cv2.imwrite(out_file, image)
            print("Image saved")
        except:
            print("error printing image")  

    except ValueError:
        print("Value error, this image will not be saved as we cannot know the frame number")     

def get_number_frames(input_video):
    '''
    This function returns the number of frames in a given video
    - input_video = absolute path of the video
    '''

    # Specify SVO path parameter
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(input_video))
    init_params.svo_real_time_mode = False  # Don't convert in realtime
    init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)

    # Create ZED objects
    zed = sl.Camera()

    # Open the SVO file specified as a parameter
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        sys.stdout.write(repr(err))
        zed.close()
        exit()
    
    nb_frames = zed.get_svo_number_of_frames()
    return(nb_frames)

def max_length_video(input_folder, max_length=500):
    '''
    This function checks whether the video is longer than the maximum length. If longer, it cuts the video to the maximum length.
    - input_folder = directory where the video is located
    - max_length = maximum length of the video
    '''
    import subprocess
    import math
    videos_to_split = dict()
    for files in os.listdir(input_folder):
        if files.endswith(".svo"):
            nb_frames = get_number_frames(os.path.join(input_folder,files))
            if nb_frames > max_length:
                videos_to_split[files] = nb_frames
    #print(videos_to_split)
    if len(videos_to_split)>0:
        print("There are videos that are longer than the maximum length")
        for file_, nb_frames in videos_to_split.items():
            #print("file is "+str(file_))
            #print("nb_frames is "+str(nb_frames))
            n_files = int(math.ceil(nb_frames/max_length))
            for i in range(n_files):
                start_frame = int(i*max_length)
                end_frame = int((i+1)*max_length)
                if end_frame > nb_frames:
                    end_frame = int(nb_frames-3)

                #print("Starting at "+str(start_frame)+" and ending at "+str(end_frame))
                subprocess.call(["\\\\gb010587mm\\Software_dev\\Ash_Dieback_Solution\\ash_dieback_solution\\models\\ZED_SVOEditor.exe","cut",os.path.join(input_folder,file_),"-s",str(start_frame),"-e",str(end_frame),os.path.join(input_folder,file_[:-4]+"_"+str(i)+".svo")])
                
        for key in videos_to_split.keys():
            os.remove(os.path.join(input_folder,key))
    else:
        print("No videos to split")

def new_to_avi(input_vid,output_video,vid_resolution):
    # Get input parameters
    svo_input_path = input_vid
    output_path = output_video
    output_as_video = True    
   
    # Specify SVO path parameter
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(svo_input_path))
    init_params.svo_real_time_mode = False  # Don't convert in realtime
    init_params.coordinate_units = sl.UNIT.METER  # Use milliliter units (for depth measurements)

    # Create ZED objects
    zed = sl.Camera()

    # Open the SVO file specified as a parameter
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        sys.stdout.write(repr(err))
        zed.close()
        exit()
    
    # Get image size
    image_size = zed.get_camera_information().camera_resolution
    camera_infos = zed.get_camera_information()
    width_HD = 1920
    height_HD = 1080
    width_sbs_HD = width_HD

    # Prepare side by side image container equivalent to CV_8UC4
    svo_image_sbs_rgba_HD = np.zeros((height_HD, width_sbs_HD, 4), dtype=np.uint8)
    
    # Get image size
    image_size = zed.get_camera_information().camera_resolution
    width_2K = 2208
    height_2K = 1242
    width_sbs_2K = width_2K

    # Prepare side by side image container equivalent to CV_8UC4
    svo_image_sbs_rgba_2K = np.zeros((height_2K, width_sbs_2K, 4), dtype=np.uint8)

    # Prepare single image containers
    left_image = sl.Mat()

    video_writer = None
    if vid_resolution == "HD":
        video_writer = cv2.VideoWriter(str(output_path),cv2.VideoWriter_fourcc('M', '4', 'S', '2'),zed.get_camera_information().camera_fps,(width_sbs_HD, height_HD))
    else:
        video_writer = cv2.VideoWriter(str(output_path),cv2.VideoWriter_fourcc('M', '4', 'S', '2'),zed.get_camera_information().camera_fps,(width_sbs_2K, height_2K))

    if not video_writer.isOpened():
        sys.stdout.write("OpenCV video writer cannot be opened. Please check the .avi file path and write "
                             "permissions.\n")
        zed.close()
        exit()
    
    rt_param = sl.RuntimeParameters()
    rt_param.sensing_mode = sl.SENSING_MODE.STANDARD

    # Start SVO conversion to AVI/SEQUENCE
    sys.stdout.write("Converting SVO... Use Ctrl-C to interrupt conversion.\n")

    nb_frames = zed.get_svo_number_of_frames()
    n_loops = 0
    n_error = 0
    while True:
        if zed.grab(rt_param) == sl.ERROR_CODE.SUCCESS:
            svo_position = zed.get_svo_position()

            # Retrieve SVO images
            zed.retrieve_image(left_image, sl.VIEW.LEFT)

            if vid_resolution == "HD":
                svo_image_sbs_rgba_HD[:, :, :] = left_image.get_data()
                # Convert SVO image from RGBA to RGB
                ocv_image_sbs_rgb = cv2.cvtColor(svo_image_sbs_rgba_HD, cv2.COLOR_RGBA2RGB)
            
            else:
                svo_image_sbs_rgba_2K[:, :, :] = left_image.get_data()
                # Convert SVO image from RGBA to RGB
                ocv_image_sbs_rgb = cv2.cvtColor(svo_image_sbs_rgba_2K, cv2.COLOR_RGBA2RGB)

            # Write the RGB image in the video
            video_writer.write(ocv_image_sbs_rgb)
           
            # Display progress
            progress_bar((svo_position + 1) / nb_frames * 100, 30)

            # Check if we have reached the end of the video
            if svo_position >= (nb_frames - 1):  # End of SVO
                sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
                break
        else:
            sys.stdout.write("\nLoop was exited but video generated\n")
            video_writer.release()
            break
        n_loops = n_loops + 1

        if ((n_loops - (svo_position)) > 100):  # End of SVO
            sys.stdout.write("\nLoop was exited but video generated\n")
            video_writer.release()
            break

        elif n_error > 10:
            sys.stdout.write("\nLoop was exited but video generated\n")
            video_writer.release()
            break
    
    video_writer.release()
    zed.close()

if __name__ == "__main__":
    folder_ = r"\\gb010587mm\Software_dev\Ash_Dieback_Solution\Tests\video_ls\original_video"
    out_folder = r"\\gb010587mm\Software_dev\Ash_Dieback_Solution\Tests\Monmouth_val"
    input_vid = r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\Surveyors_SD_Card_Data\Monmouth\Monmouth\Part 3\ZED\video_front_cam_14-07_14-57-25.svo"

    new_to_avi(input_vid=input_vid,output_video=os.path.join(out_folder,"monmouth_val"+".avi"),vid_resolution="2K")


