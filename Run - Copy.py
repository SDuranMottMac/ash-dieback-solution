## Package information
# ==================
__author__ = ["Sergio Duran"] # Please, include your name if you have contributed to develop this package
__License__ = "Mott MacDonald"
__maintainer__ = ["Sergio Duran"] # Please, include your name if you have contributed to develop this package
__email__ = ["sergio.duranalvarez@mottmac.com"] # Please, include your email if you have contributed to develop this package
__status__ = "Development"
__version__ = "0.0.1"

# importing libraries
# ===================
import argparse
from genericpath import isfile
import albumentations as A
import cv2
import json
import logging
import numpy as np
import os
from pathlib import Path
import pyzed.sl as sl
import torch.backends.cudnn as cudnn
import shutil
from threading import Lock, Thread, Event
import threading
import _thread
from tensorflow import keras
import time
from time import sleep
import torch
import random
import sys
import uuid
from contextlib import contextmanager
import multiprocessing
# releasing memory
# from numba import cuda


# importing Zed dependencies
# ===========================
sys.path.insert(1, './src/pipeline')
import cv_viewer.tracking_viewer as cv_viewer
from image_preprocessing import img_preprocess,xywh2abcd,bounding_boxes,drawing_bbox,drawing_bbox_and_distance,detections_to_custom_box,new_detections_to_custom_box,get_center_box,generate_YOLO_label
from mti_reader import generating_timestamp,collecting_mtidata,ts_to_coordinates,ts_to_rear_coordinates,last_modified_file,check_loop,estimating_abs_height,collecting_mtidata_pitch,collecting_mtidata_pitch_lessfreq,strip_mt_data,split_mt_data
import ogl_viewer.viewer as gl
from video_generation import progress_bar, turning_video_oneimage,max_length_video, new_to_avi
from shapefile_generator import shapefile_from_dict,merging_shp
from email_sender import Mail
from coordinate_estimator import inverse_distance, last_estimations
from coordinate_estimator import inverse_distance, last_estimations,last_estimations_no_outliers,last_rear_estimations_no_outliers
from blur_faces import blurBoxes, blur_faces
from deepsort_tracker import detect_and_track,reordering_deepsort_corrected,identify_smaller
from NTM_post_process import auto_processing

# importing Yolo dependencies
# ===========================
sys.path.insert(0, './models/Object_detection/yolov5')
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.torch_utils import select_device
from utils.augmentations import letterbox

# importing Timm dependencies
# ===========================
sys.path.insert(2, './models/Image_classification/pytorch-image-models')
from get_inference import inference_classification,get_fp_assessment,dieback_batch_class,dieback_class_Resnet50

# importing additional dependenies
# ================================
sys.path.insert(3, './src/additional')
from duplicates_videos import DuplicateID


# Initiating Threading
lock = Lock()
run_signal = False
exit_signal = False

# List of mails where errors will be sent
mails = ["Sergio.DuranAlvarez@mottmac.com","tom.doughty@mottmac.com","john.farrow@mottmac.com"]

# Adding extra functionality
# =========================
def torch_thread(weights, img_size, conf_thres=0.2, iou_thres=0.45):
    """
    This thread is responsible to run the Yolo algorithm when it is required.
    """
    # Defining global variables - these variables will be shareb between different sections of the code
    global image_net, exit_signal, run_signal, detections
    # Logging initialization message
    logging.info("Intializing Network...")
    # Selecting device where the model runs
    device = select_device()
    # Halving the precision if using CUDA
    half = device.type != 'cpu' 
    # Defining the image size 
    imgsz = img_size

    # Load model - FP32
    model = attempt_load(weights, device=device)
    # Defining the strides of the model  
    stride = int(model.stride.max())
    # Checking whether the strides + filters are compatible with the image size 
    imgsz = check_img_size(imgsz, s=stride) 
    # If we use CUDA, we use half precision
    if half:
        model.half() 
    # Enabling the cudnn benchmark
    """
    Note: benchmark mode is good whenever your input sizes for your network do not vary. 
    This way, cudnn will look for the optimal set of algorithms for that particular configuration (which takes some time). 
    This usually leads to faster runtime.
    """
    cudnn.benchmark = True
    
    # Run inference
    if (device.type != 'cpu'):
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  

    # Looping until the exit signal is activated
    while not exit_signal:
        # if run signal is activated - the yolo model makes predictions 
        if run_signal:
            # Acquiring a lock
            lock.acquire()
            # Preprocessing the image
            img, ratio, pad = img_preprocess(image_net, device, half, imgsz) 
            # Obtaining predictions          
            pred = model(img)[0]          
            # Max suppression and filtering by class 
            det = non_max_suppression(pred, conf_thres, iou_thres,classes=[0,2])    
            # ZED CustomBox format (with inverse letterboxing tf applied)          
            detections = new_detections_to_custom_box(det, img, image_net) 
            # Releasing the lock
            lock.release()
            # Disabling the run signal
            run_signal = False
            # Waiting for a few ms so we do not overwhelm the computer while
            sleepy_time = random.uniform(0.2,1)
            sleepy_time = sleepy_time/1000
            sleep(sleepy_time)

        # We sleep the loop again
        sleepy_time = random.uniform(0.2,1)
        sleepy_time = sleepy_time/100
        sleep(sleepy_time)
    
    print("Thread finished")
    # Thread terminates
    sys.exit()


class TimeoutException(Exception):
    def __init__(self, msg=''):
        self.msg = msg

def main():
    """
    This method defines the main stream of the pipeline - from initiating the cameras to returning predictions.
    """
    # First, we are going to define the global variables to be shared with other methods
    global image_net, exit_signal, run_signal, detections, temp_dir, deepsort_dict, duplicate_data_list, n_frames_processed
    
    # The Torch thread defined above starts
    exit_signal = False
    capture_thread = Thread(target=torch_thread, kwargs={'weights': opt.weights, 'img_size': opt.img_size, "conf_thres": opt.conf_thres})
    capture_thread.start()
    
    # Initiating the Camera
    logging.info("Initializing Camera...")
    zed = sl.Camera()
    input_type = sl.InputType()

    '''
    As we are going to be providing 2 different types of inputs (folders with .svo files or a .svo file directly),
    we need to define this accordingly
    '''

    if opt.svo is not None:
        if os.path.isdir(opt.svo):
            input_type.set_from_svo_file(svo_video)
        else:
            input_type.set_from_svo_file(opt.svo)

    # Create a InitParameters object and set configuration parameters
    '''
    To configure the depth sensing, we use InitParameters at initialization and RuntimeParameters to change specific parameters during use.
    Basically,
    initParam - Holds the options used to initialised the Camera object
    runtimeParam - Parameters that defines the behaviour of the grab.
    '''
    
    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=opt.real_time)
    if opt.resolution == "HD":
        init_params.camera_resolution = sl.RESOLUTION.HD1080
    else:
        init_params.camera_resolution = sl.RESOLUTION.HD2K
    
    # Setting the frequency of the camera 
    init_params.camera_fps = max(zed.get_camera_information().camera_fps, 15)
    # Setting the unit to be used in the estimations 
    init_params.coordinate_units = sl.UNIT.METER 
    # Setting the depth mode - ULTRA: offers the highest depth range and better preserves Z-accuracy along the sensing range
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    # Setting the coordinate system
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    # Setting the maximum depth distance
    init_params.depth_maximum_distance = int(opt.dist_thres)
    # Setting the minimum depth distance  
    init_params.depth_minimum_distance = 1 

    # Defining the run time parameters
    runtime_params = sl.RuntimeParameters()
    # Setting standard sensing mode - preserves edges and depth accuracy
    runtime_params.sensing_mode = sl.SENSING_MODE.STANDARD
    # Setting a confidence threshold to reject depth values
    runtime_params.confidence_threshold = 10

    # Retrieving the status of the instantiation
    status = zed.open(init_params)

    # asserting the status is SUCCESS
    if status != sl.ERROR_CODE.SUCCESS:
        logging.warning(repr(status))
        exit()

    # Instantiating a matrix object with the characteristics of the image
    image_left_tmp = sl.Mat()

    logging.info("Initialized Camera")

    # positional tracking and detection parameters
    '''
    Positional parameters - These are parameters for tracking initialisation
    ObjectDetection - object detection parameters
    Objects - Contains the result of the object detection module
    '''
    # Instantiating a positional tracking object
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # set floor as origin
    positional_tracking_parameters.set_floor_as_origin = True
    # Enabling positional tracking
    # zed.enable_positional_tracking(positional_tracking_parameters)
    if zed.enable_positional_tracking(positional_tracking_parameters) != sl.ERROR_CODE.SUCCESS:
        logging.warning("Error while enabling positional tracking")
        exit()
    
    # Instantiating an object detection parameters
    obj_param = sl.ObjectDetectionParameters()
    # Setting the custom box object as bounding box object
    obj_param.detection_model = sl.DETECTION_MODEL.CUSTOM_BOX_OBJECTS 
    # Enable the intern tracker
    obj_param.enable_tracking = True
    # setting the object detection module to run asynchronously - thus, retrieve_objects does not get stuck
    obj_param.image_sync = False
    # Set the object detector with the defined attributes
    # zed.enable_object_detection(obj_param)
    if zed.enable_object_detection(obj_param) != sl.ERROR_CODE.SUCCESS:
        logging.warning("Error while enabling object detection")
        exit()

    # Instantiate Object
    objects = sl.Objects()
    # Instantiating a runtime parameters object
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    # Defining the confidence threshold for the object detector
    obj_runtime_param.detection_confidence_threshold = 1
    # Filtering by class - not currently enabled as Yolo will do it
    obj_runtime_param.object_class_filter = {} # select which object types to detect and track
    # Display options
    camera_infos = zed.get_camera_information()
    # Create OpenGL viewer if enabled
    if opt.viewer == True:
        viewer = gl.GLViewer()
        # Define the resolution for the point cloud
        point_cloud_res = sl.Resolution(min(camera_infos.camera_resolution.width, 720),
                                        min(camera_infos.camera_resolution.height, 404))
        point_cloud_render = sl.Mat()
    # initiate the viewer if enabled
    if opt.viewer == True:
        viewer.init(camera_infos.camera_model, point_cloud_res, obj_param.enable_tracking)
        point_cloud = sl.Mat(point_cloud_res.width, point_cloud_res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
        image_left = sl.Mat()
    
    # Utilities for 2D display
    if opt.viewer == True:
        # define the display resolution (width/height)
        display_resolution = sl.Resolution(min(camera_infos.camera_resolution.width, 1280),
                                        min(camera_infos.camera_resolution.height, 720))
        # scaling between the display resolution and the camera resolution
        image_scale = [display_resolution.width / camera_infos.camera_resolution.width, display_resolution.height / camera_infos.camera_resolution.height]
        # and create a np array with the values provided for the image left ocv
        image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4), [245, 239, 239, 255], np.uint8)

        # Utilities for tracks view
        camera_config = zed.get_camera_information().camera_configuration
        tracks_resolution = sl.Resolution(400, display_resolution.height)
        track_view_generator = cv_viewer.TrackingViewer(tracks_resolution, camera_config.camera_fps,
                                                        init_params.depth_maximum_distance)
        track_view_generator.set_camera_calibration(camera_config.calibration_parameters)
        image_track_ocv = np.zeros((tracks_resolution.height, tracks_resolution.width, 4), np.uint8)
        # Camera pose
        cam_w_pose = sl.Pose()

    # saving the scenes
    '''
    As part of the pipeline, we save all the scenes where the pipeline evaluated whether there is any detections.
    These scenes will be used later on for
        - creating images with bounding boxes (for further assessment)
        - creating images without bounding boxes (for delivery)
        - creating cropped images for health classification
    Note - the number of scenes does not neccesarily match the number of frames
    '''
    if os.path.isdir(opt.svo): 
        if not os.path.isdir(temp_dir):
            os.makedirs(temp_dir)
    else:
        path = Path(opt.svo)
        temp_dir = path.parent.absolute()
        os.makedirs(temp_dir)

    # A detection dictionary is created where the detections are saved
    detections_dict = dict()
    # Defining the starting frame number
    frame_number = -1
    # Creating a temporary folder where the images with detections will be saved
    if not os.path.isdir(os.path.join(outpath,"Cropped")):
        os.makedirs(os.path.join(outpath,"Cropped")) 
    # we create a folder for monitoring the model in production
    if not os.path.isdir(os.path.join(outpath,"Monitoring")):
        os.makedirs(os.path.join(outpath,"Monitoring"))
        os.makedirs(os.path.join(outpath,"Monitoring","Images"))
        os.makedirs(os.path.join(outpath,"Monitoring","Labels"))
    
    # creating a json file to save the detection dictionary
    # checking whether it already exists
    if os.path.isfile(os.path.join(opt.svo,"detections_processed.json")):
        # if file exists, read and load in the detection dict
        with open(os.path.join(opt.svo,"detections_processed.json"),"r") as json_file_processed:
            detections_dict = json.load(json_file_processed)
    # else, just create the empty json to be populated
    else:
        with open(os.path.join(opt.svo,"detections_processed.json"),"w") as json_file_processed:
            json.dump(detections_dict, json_file_processed)

    # Starting the loop
    while not exit_signal:

        # Add some waiting time
        time.sleep(0.001)
        # checking whether there are frames already processed
        """
        In the event of resuming the process, this will loop over the already processed frames up to the place where it stopped
        """
        if (int(frame_number+1) < int(n_frames_processed)):
            print("Frame number "+str(frame_number+1)+" and the frame processed previously are "+str(n_frames_processed))
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                # obtaining frame number
                frame_number = frame_number + 1
                # -- Get the image/scene
                lock.acquire()
                # Retrieving the left image and saving into the left tmp matrix
                if zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT) != sl.ERROR_CODE.SUCCESS:
                    # the Lock is released and the next image is grabbed
                    lock.release()
                    continue
                # zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
                # Getting the data out of the matrix and saving in image_net
                image_net = image_left_tmp.get_data()
                # Convert SVO image from RGBA to RGB
                ocv_image_sbs_rgb = cv2.cvtColor(image_net, cv2.COLOR_RGBA2RGB)
                # the Lock is released
                lock.release()
                # adding some waiting time to avoid overwhelming the process
                time.sleep(0.001)
                # grabbing the next image
                continue
            else:
                # obtaining frame number
                frame_number = frame_number + 1
                continue

        # Processing the yet to be procesed part of the video
        elif zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # obtaining frame number
            frame_number = frame_number + 1
            # checking whether frame number concurs with svo position
            if int(frame_number) != int(zed.get_svo_position()):
                frame_number = int(zed.get_svo_position())
            # -- Get the image/scene
            lock.acquire()
            # Retrieving the left image and saving into the left tmp matrix
            if zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT) != sl.ERROR_CODE.SUCCESS:
                # the Lock is released and the next image is grabbed
                lock.release()
                continue
            # Getting the data out of the matrix and saving in image_net
            image_net = image_left_tmp.get_data()
            # Convert SVO image from RGBA to RGB
            ocv_image_sbs_rgb = cv2.cvtColor(image_net, cv2.COLOR_RGBA2RGB)
            # Saving the scene in the scenes folder
            cv2.imwrite(os.path.join(temp_dir,str(frame_number)+".png"),ocv_image_sbs_rgb)
            # the Lock is released
            lock.release()
            
            # Run signal variable is activated
            run_signal = True
            # -- Detection running on the other thread
            """
            The other thread will predict the current image and disable the variable run signal.
            """
            while run_signal:
                sleepy_time2 = random.uniform(0,1)
                sleepy_time2 = sleepy_time2/1000
                sleep(sleepy_time2)   

            # check whether there are detections - if no detection, skip it and continue
            if not detections:
                logging.info("No detections in this image")
                continue
            # Adding some waiting time to the process
            time.sleep(0.001)  
            
            # Initiate another lock
            lock.acquire()
            # -- Ingest detections
            # zed.ingest_custom_box_objects(detections)
            if zed.ingest_custom_box_objects(detections) != sl.ERROR_CODE.SUCCESS:
                logging.info("Error while ingesting the bbox")
                lock.release()
                continue
            else:
                lock.release()
            # Adding some waiting time to the process
            time.sleep(0.001)

            # Retrieving the detections       
            # If it works, we retrieve the objects
            # print("About to retrieve objects")
            # if zed.retrieve_objects(objects, obj_runtime_param) != sl.ERROR_CODE.SUCCESS:
            #     print("Error while retrieving object")
            #     continue
            # else:
            #     print("Retrieved objects")
            #     time.sleep(0.002)

            # Move to RetrievalStarted
            with open(os.path.join(opt.svo,'state.data'), 'w') as f:
                f.write('RetrievalStarted')

            # Retrieving the detections       
            # If it works, we retrieve the objects
            print("About to retrieve objects")
            if zed.retrieve_objects(objects, obj_runtime_param) != sl.ERROR_CODE.SUCCESS:
                print("Error while retrieving object")
                # Move to RetrievalComplete
                with open(os.path.join(opt.svo,'state.data'), 'w') as f:
                    f.write('RetrievalComplete')
                continue
            else:
                print("Retrieved objects")
                time.sleep(0.002)

            # Move to RetrievalCompleteos.remove
            with open(os.path.join(opt.svo,'state.data'), 'w') as f:
                f.write('RetrievalComplete')

            # -- Display
            # Retrieve display data
            if opt.viewer == True:
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, point_cloud_res)
                point_cloud.copy_to(point_cloud_render)
                zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
                zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)
            # 3D rendering
            if opt.viewer == True:
                viewer.updateData(point_cloud_render, objects)
                # 2D rendering
                np.copyto(image_left_ocv, image_left.get_data())
                cv_viewer.render_2D(image_left_ocv, image_scale, objects, obj_param.enable_tracking)
                global_image = cv2.hconcat([image_left_ocv, image_track_ocv])
                # Tracking view
                track_view_generator.generate_view(objects, cam_w_pose, image_track_ocv, objects.is_tracked)
                cv2.imshow("ZED | 2D View and Birds View", global_image)
                key = cv2.waitKey(10)
            
            # saving all the information
            '''
            All the information generated during this process is saved into a dictionary, which is subsequently exported as a json file.
            This dictionary/file can be used for coordinate calculation and other operations.
            '''
            # -- Monitoring
            # -----------------
            # Assigning a low probability
            monitoring_number = np.random.uniform(low=0, high=630) 
            monitoring_percentage = 2 
            unique_identifier = uuid.uuid4().hex[:6].upper()
            lbl_name = unique_identifier+"_monitoring_data"+str(frame_number)+".txt"
            lbl_path = os.path.join(outpath,"Monitoring","Labels",lbl_name)
            # Getting all the detections in the scene
            if objects.object_list:
                detected_objects = []
                for object in objects.object_list:
                    if (abs(object.position[2] > abs(opt.dist_thres))):
                        detected_objects.append(object)
                # If there are detections, evaluating whether the image is saved for monitoring
                if detected_objects:
                    # Evaluating whether this image is saved
                    if (float(monitoring_number) <= float(monitoring_percentage)):
                        scene = os.path.join(temp_dir,str(frame_number)+".png")
                        img_name = unique_identifier+"_monitoring_data"+str(frame_number)+".png"
                        shutil.copy(scene, os.path.join(outpath,"Monitoring","Images",img_name))
                        with open(lbl_path, 'w') as fp:
                            pass
            
            # Looping over all the detected object 
            deepsort_used = [] 
           
            for object in objects.object_list:
                '''
                The Objects class stores all the information regarding the different objects present in the scene in it object_list attribute. 
                Each individual object is stored as a ObjectData with all information about it, such as bounding box, position, mask, etc.
                All objects from a given frame are stored in a vector within Objects. Objects also contains the timestamp of the detection, 
                which can help connect the objects to the images 
                '''
                # Evaluating whether it corresponds to a duplicate frame
                timestamp =round(objects.timestamp.get_nanoseconds()/1000000000,2)+float(opt.time_drift)
                if timestamp in duplicate_data_list:
                    logging.info("Duplicate frame skipped. Timestamp: "+str(timestamp))
                    break
                # Evaluating whether the detected object meets the distance and confidence criteria
                if ((abs(object.position[2]) < abs(float(opt.dist_thres)))):
                    # check if this image has been selected for monitoring
                    if (float(monitoring_number) <= float(monitoring_percentage)):
                        # we save the label and image
                        scene = os.path.join(temp_dir,str(frame_number)+".png")
                        original_image= cv2.imread(scene)
                        img_shape = (original_image.shape[1], original_image.shape[0])
                        label_to_write = generate_YOLO_label(bbox=object.bounding_box_2d.tolist(),class_number=1,img_shape=img_shape,score=object.confidence)
                        with open(lbl_path, 'a') as fp:
                            fp.write(label_to_write)
                    # check if this detection has a match with deepsort
                    id_match = []
                    dist_match = []
                    conf_assoc = []
                    # If DeepSort has no predictions for this frame - continue
                    if not str(frame_number) in deepsort_dict.keys():
                        continue
                    # Evaluating which ID is the match
                    for idxs in range(len(deepsort_dict[str(frame_number)]["id"])):
                        # obtaining reference point coordinates
                        cropping_box = object.bounding_box_2d.tolist()
                        y =int(min(cropping_box[0][1],cropping_box[1][1],cropping_box[2][1],cropping_box[3][1]))
                        h = int(max(cropping_box[0][1],cropping_box[1][1],cropping_box[2][1],cropping_box[3][1]) - min(cropping_box[0][1],cropping_box[1][1],cropping_box[2][1],cropping_box[3][1]))
                        x =int(min(cropping_box[0][0],cropping_box[1][0],cropping_box[2][0],cropping_box[3][0]))
                        w = int(max(cropping_box[0][0],cropping_box[1][0],cropping_box[2][0],cropping_box[3][0]) - min(cropping_box[0][0],cropping_box[1][0],cropping_box[2][0],cropping_box[3][0]))
                        #compute euclidean distance
                        reference_point = np.array([int(int(x)+int(w/2)),int((int(y)+int(h/2)))])
                        compared_point = np.array([int(int(deepsort_dict[str(frame_number)]["bbox_left"][idxs])+int(int(deepsort_dict[str(frame_number)]["bbox_w"][idxs])/2)),int(int(deepsort_dict[str(frame_number)]["bbox_top"][idxs])+int(int(deepsort_dict[str(frame_number)]["bbox_h"][idxs])/2))])
                        dist =  np.linalg.norm(reference_point-compared_point)

                        if opt.resolution == "HD":
                            if (dist < 100):
                                id_match.append(str(deepsort_dict[str(frame_number)]["id"][idxs]))
                                dist_match.append(dist)
                                conf_assoc.append(float(deepsort_dict[str(frame_number)]["conf"][idxs]))
                        else:
                            if (dist < 200):
                                id_match.append(str(deepsort_dict[str(frame_number)]["id"][idxs]))
                                dist_match.append(dist)
                                conf_assoc.append(float(deepsort_dict[str(frame_number)]["conf"][idxs]))
                    
                    object_id = "Not found"
                    # If there is no match, we use the internal tracker
                    if len(id_match) == 0:
                        logging.info("No match with deepsort")
                        continue # for now, we'll skip all detections that do not match with deepsort
                        seed_detection = 8754
                        associated_id = int(object.id)+seed_detection
                        object_id = str(associated_id)
                        object_conf = float(round(random.uniform(20,35),0))
                        logging.info("ID from internal tracker provided: "+str(object_id))
                    # Otherwise, we use the closest object detected
                    else:
                        min_index = identify_smaller(dist_list=dist_match)
                        potential_object_id = id_match[min_index]
                        # if the closest ID has not already been used by another object in the scene, we use it
                        if potential_object_id not in deepsort_used:
                            object_id = str(potential_object_id)
                            object_conf = float(conf_assoc[min_index])
                            deepsort_used.append(object_id)
                        else:
                            # otherwise, we loop over all the close objects
                            for ix in range(len(id_match)):
                                if id_match[ix] in deepsort_used:
                                    continue
                                else:
                                    # if one of them has not been used, we use it
                                    object_id = id_match[ix]
                                    object_conf = float(conf_assoc[ix])
                                    deepsort_used.append(object_id)
                    # in case of not finding objects, we provide the tracker ID 
                    if (object_id == "Not found"):
                        continue # for now, we'll skip all detections that do not match with deepsort
                        seed_detection = 8754
                        associated_id = int(object.id)+seed_detection
                        object_id = str(associated_id)
                        object_conf = float(round(random.uniform(20,35),0))

                    # we Save the info in the detection dictionary, using the ID as key
                    # If this detection has already appeared in previous frames, we append the info
                    if (object_id in detections_dict):
                        # get timestamp in correct magnitude
                        timestamp =round(objects.timestamp.get_nanoseconds()/1000000000,2)+float(opt.time_drift)
                        # if not duplicate, carry on
                        detections_dict[object_id]["height"].append(abs(object.dimensions[1]))
                        detections_dict[object_id]["distance"].append(abs(object.position[2]))
                        detections_dict[object_id]["b_box"].append(object.bounding_box_2d.tolist())
                        detections_dict[object_id]["timestamp"].append(timestamp)
                        detections_dict[object_id]["frame"].append(os.path.join(temp_dir,str(frame_number)+".png"))
                        detections_dict[object_id]["score"].append(object_conf)
                        # Dealing with image classification
                        logging.info("Generating cropped images")
                        # 1 - reading image
                        scene = os.path.join(temp_dir,str(frame_number)+".png")
                        original_image= cv2.imread(scene)
                        # 2 - generating the cropping box
                        cropping_box = object.bounding_box_2d.tolist()
                        y =int(min(cropping_box[0][1],cropping_box[1][1],cropping_box[2][1],cropping_box[3][1]))
                        h = int(max(cropping_box[0][1],cropping_box[1][1],cropping_box[2][1],cropping_box[3][1]) - min(cropping_box[0][1],cropping_box[1][1],cropping_box[2][1],cropping_box[3][1]))
                        x =int(min(cropping_box[0][0],cropping_box[1][0],cropping_box[2][0],cropping_box[3][0]))
                        w = int(max(cropping_box[0][0],cropping_box[1][0],cropping_box[2][0],cropping_box[3][0]) - min(cropping_box[0][0],cropping_box[1][0],cropping_box[2][0],cropping_box[3][0]))
                        # calculate coordinates
                        x_center = float(float(x) + (float(w)/2))
                        y_center = float(float(y) + (float(h)/2))
                        # estimating latitude longitude
                        if opt.rear_videos:
                            detection_longitude, detection_latitude = ts_to_rear_coordinates(ts=timestamp,mit_data=mti_data,distance=abs(object.position[2]),
                                img_with=camera_infos.camera_resolution.width,center_x=x_center)
                        else:
                            detection_longitude, detection_latitude = ts_to_coordinates(ts=timestamp,mit_data=mti_data,distance=abs(object.position[2]),
                                img_with=camera_infos.camera_resolution.width,center_x=x_center)
                        # checking whether the timestamp was found
                        if (detection_longitude == None) or (detection_latitude == None):
                            continue
                        # append the coordinates information
                        detections_dict[object_id]["detection_latitude"].append(detection_latitude)
                        detections_dict[object_id]["detection_longitude"].append(detection_longitude)
                        # 3- generating the cropping image
                        cropped_image = original_image[y:y+h,x:x+w]
                        # - Estimating absolute height
                        if timestamp in mti_data:
                            #print("Found timestamp")
                            pitch_angle = mti_data[timestamp]["Pitch"]
                            elev_car = mti_data[timestamp]["Altitude"]
                        else:
                            #print("No TS found")
                            pitch_angle = 0
                            elev_car = 0
                        try:
                            abs_height = estimating_abs_height(distance=abs(object.position[2]), elev_car=elev_car, height_pix=h, height_real=abs(object.dimensions[1]), bottom_y=int(max(cropping_box[0][1],cropping_box[1][1],cropping_box[2][1],cropping_box[3][1])),pitch_angle=pitch_angle,height_image=camera_infos.camera_resolution.height)
                        except:
                            abs_height = int(0)
                        detections_dict[object_id]["abs_height"].append(abs_height)
                        # 4 - saving the cropped image
                        try:
                            # 5 - image number
                            n_images = len(os.listdir(os.path.join(outpath,"Cropped",str(object_id)))) + 1
                            # 6 - image name
                            img_name = "image_"+str(n_images)+".png"
                            # Applying transformations
                            transform = A.Compose([
                                A.LongestMaxSize(max_size=224,interpolation=1),
                                A.PadIfNeeded(min_height=224,min_width=224,border_mode=2)
                                ])
                            transformed = transform(image=cropped_image)
                            transformed_image = transformed["image"]
                            # 7 - saving the image in the corresponding folder
                            cv2.imwrite(os.path.join(outpath,"Cropped",str(object_id),img_name),transformed_image)
                            logging.info("Cropped image "+str(n_images)+" from id "+str(object_id)+" has been saved")
                        except:
                            logging.info("image skipped")

                    # If that's the first time that the ID appears
                    else:
                        # get timestamp in correct magnitude
                        timestamp =round(objects.timestamp.get_nanoseconds()/1000000000,2)+float(opt.time_drift)
                        # estimate dieback level
                        logging.info("Generating cropped images")
                        # reading image

                        scene = os.path.join(temp_dir,str(frame_number)+".png")
                        original_image= cv2.imread(scene)
                        # generating the cropping box
                        cropping_box = object.bounding_box_2d.tolist()
                        y =int(min(cropping_box[0][1],cropping_box[1][1],cropping_box[2][1],cropping_box[3][1]))
                        h = int(max(cropping_box[0][1],cropping_box[1][1],cropping_box[2][1],cropping_box[3][1]) - min(cropping_box[0][1],cropping_box[1][1],cropping_box[2][1],cropping_box[3][1]))
                        x =int(min(cropping_box[0][0],cropping_box[1][0],cropping_box[2][0],cropping_box[3][0]))
                        w = int(max(cropping_box[0][0],cropping_box[1][0],cropping_box[2][0],cropping_box[3][0]) - min(cropping_box[0][0],cropping_box[1][0],cropping_box[2][0],cropping_box[3][0]))
                        # generating the cropping image
                        cropped_image = original_image[y:y+h,x:x+w]
                        # calculate coordinates
                        x_center = float(float(x)+(float(w)/2))
                        y_center = float(float(y)+(float(h)/2))
                        # estimating latitude longitude
                        if opt.rear_videos:
                            detection_longitude, detection_latitude = ts_to_rear_coordinates(ts=timestamp,mit_data=mti_data,distance=abs(object.position[2]),
                                img_with=camera_infos.camera_resolution.width,center_x=x_center)
                        else:
                            detection_longitude, detection_latitude = ts_to_coordinates(ts=timestamp,mit_data=mti_data,distance=abs(object.position[2]),
                                img_with=camera_infos.camera_resolution.width,center_x=x_center)
                        # checking whether the timestamp was found
                        if (detection_longitude == None) or (detection_latitude == None):
                            continue
                        # - Estimating absolute height
                        if timestamp in mti_data:
                            logging.info("Found timestamp. Extracting pitch and altitude")
                            pitch_angle = mti_data[timestamp]["Pitch"]
                            elev_car = mti_data[timestamp]["Altitude"]
                        else:
                            logging.info("No Timestamp. Pitch and elevation set as 0")
                            pitch_angle = 0
                            elev_car = 0
                        try:
                            abs_height = estimating_abs_height(distance=abs(object.position[2]), elev_car=elev_car, height_pix=h, height_real=abs(object.dimensions[1]), bottom_y=int(max(cropping_box[0][1],cropping_box[1][1],cropping_box[2][1],cropping_box[3][1])),pitch_angle=pitch_angle,height_image=camera_infos.camera_resolution.height)
                        except:
                            abs_height = int(0)
                        # saving the cropped image
                        try:
                            # creating a directory for this ID
                            os.makedirs(os.path.join(outpath,"Cropped",str(object_id)))
                            # generating image name
                            img_name = "image_"+str(1)+".png"
                            try:
                                transform = A.Compose([
                                    A.LongestMaxSize(max_size=224,interpolation=1),
                                    A.PadIfNeeded(min_height=224,min_width=224,border_mode=2)
                                    ])
                                transformed = transform(image=cropped_image)
                                transformed_image = transformed["image"]    
                                # saving the image
                                cv2.imwrite(os.path.join(outpath,"Cropped",str(object_id),img_name),transformed_image)
                                logging.info("Image "+str(1)+" for id "+str(object_id)+" saved")
                            except:
                                logging.info("Having problems saving cropped image")

                            detections_dict[object_id]={"height":[abs(object.dimensions[1])],"distance":[abs(object.position[2])],"frame":[os.path.join(temp_dir,str(frame_number)+".png")],"b_box":[object.bounding_box_2d.tolist()],
                                "timestamp":[timestamp],"detection_latitude":[detection_latitude],"detection_longitude":[detection_longitude],"ash_dieback":[],"abs_height":[abs_height],"score":[object_conf]}
                    
                        except:
                            logging.info("image skipped")

                    # adding data to the detection json
                    with open(os.path.join(opt.svo,"detections_processed.json"), 'w') as json_file_processed:
                        json.dump(detections_dict, json_file_processed)                       
        else:
            exit_signal = True
    if opt.viewer == True:
        viewer.exit()
    exit_signal = True
    zed.close()  
    logging.info("Finished while loop")
    # removing the json dictionary as we do not need it anymore
    os.remove(os.path.join(opt.svo,"detections_processed.json"))
    # closing the thread
    capture_thread.join()
    # returning the dictionary as variable
    return(detections_dict)

# Other minor functions
def Average_list(lst):
    if not isinstance(lst,list):
        return(lst)
    elif len(lst) == 1:
        return(lst[0])
    else:
        return sum(lst) / len(lst)

def Average_list_lists(lst):
    new_list = []
    for item in lst:
        new_list.append(item[0])
    
    if len(new_list) == 1:
        return(new_list[0])
    else:
        return(sum(new_list) / len(new_list))

# The script
# ===========
if __name__ == '__main__':

    # Defining the parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=['./weights/yolov5/yolos_1440.keras','./weights/yolov5/yolo_1440.keras'], help='model.pt path(s)')
    # parser.add_argument('--weights', nargs='+', type=str, default='./weights/yolov5/yolos_1440.keras', help='model.pt path(s)')
    parser.add_argument('--weights_class', nargs='+', type=str, default=['./weights/Resnet_50/Health_class_2.keras','./weights/Resnet_50/Health_class_4.keras'], help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file or folder')
    parser.add_argument('--resolution', type=str, default="HD", help='Resolution of the video')
    parser.add_argument('--img_size', type=int, default=1440, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.20, help='object confidence threshold')
    parser.add_argument('--out_results', type=float, default=None, help='Output where results will be generated')
    parser.add_argument('--dist_thres', type=int, default=20, help='Distance threshold for the detections')
    parser.add_argument('--viewer', action="store_true")
    parser.add_argument('--real_time', action="store_true")
    parser.add_argument('--rear_videos', action="store_true")
    parser.add_argument('--time_drift', type=int, default=0, help='adapts the pipeline to correct a time drift (+ means camera is behind)')
    opt = parser.parse_args()

    start = time.time()

    
    # Adding flexibility - we can provide either a folder with videos or a video
    if os.path.isdir(opt.svo):

        # instantiating the temporary directory
        temp_dir = os.path.join(opt.svo,"scenes")

        # Initiating logging
        # ==================
        filename_logs = os.path.join(opt.svo,"runtime.log")
        logging.basicConfig(filename=filename_logs, level=logging.INFO, filemode="a",
            format='%(levelname)s:%(message)s')

        logging.info("I have been provided a folder with videos")

        # Obtaining duplicate lists
        # =========================
        logging.info("Obtaining duplicates")
        if os.path.isfile(os.path.join(opt.svo,"duplicate_list.txt")):
            # read the file 
            duplicate_file = open(os.path.join(opt.svo,"duplicate_list.txt"), "r")
            data_duplicates = duplicate_file.read()
            # Replacing end of line by spaces and splitting by spaces
            duplicate_data_list = data_duplicates.replace('\n', ' ').split(' ')
            # closing the file
            duplicate_file.close() 

        # Else, generate the list and save it into a duplicate list
        else:
            # generate the file
            duplicate_checker = DuplicateID(input_folder=opt.svo,output_folder=opt.svo)
            duplicate_checker.obtaining_duplicatesTS()
            # read file
            # read the file 
            duplicate_file = open(os.path.join(opt.svo,"duplicate_list.txt"), "r")
            data_duplicates = duplicate_file.read()
            # Replacing end of line by spaces and splitting by spaces
            duplicate_data_list = data_duplicates.replace('\n', ' ').split(' ')
            # closing the file
            duplicate_file.close() 
        
        # the duplicates file is removed, as it will only introduce potential problems
        os.remove(os.path.join(opt.svo,"duplicate_list.txt"))

        # transform duplicate list into float numbers
        duplicate_data_list = [float(elm) for elm in duplicate_data_list if elm != '']
        
        logging.info("The list of duplicates have been obtained.")

        # Obtaining MTi Info
        # ==================
        '''
        MTi_data is a dictionary where the timestamps are the key and the values are the latitude, longitude and altitude
        '''
        # First, we need to prepare the file
        strip_mt_data(input_directory=opt.svo)
        split_mt_data(input_dir=opt.svo) 
        # Now, we need to process the file
        mti_data = collecting_mtidata_pitch_lessfreq(input_directory=opt.svo)
        #save dict to json
        logging.info("MTi info read")

        # Processing the videos
        # =====================
        for item in os.listdir(opt.svo):
            # selecting only those rear videos
            if item.endswith(".svo") and item.split("_")[1]=="rear" and opt.rear_videos:
                # The Try except will catch the error - Thus we get notified
                try:
                    logging.info("dealing with video "+str(item))
                    # we generate the absolute path of the video
                    svo_video = os.path.join(opt.svo,item)
                    # we need to make a directory for the output, so we can save the video and the images
                    if opt.out_results is None:
                        path1 = Path(svo_video)
                        if not os.path.exists(os.path.join(path1.parent,item[:-4])):
                            logging.info("Creating a new directory to store the data from "+str(item))
                            os.makedirs(os.path.join(path1.parent,item[:-4]))
                            outpath = os.path.join(path1.parent,item[:-4])
                            out_video = os.path.join(path1.parent,item[:-4],item[:-4]+".avi")
                            n_frames_processed = 0
                        elif os.path.exists(os.path.join(path1.parent,item[:-4],"Cropped")):
                            if os.path.isdir(temp_dir):
                                n_frames_processed = len(os.listdir(temp_dir))
                                outpath = os.path.join(path1.parent,item[:-4])
                                out_video = os.path.join(path1.parent,item[:-4],item[:-4]+".avi")
                            else:
                                n_frames_processed = 0
                                outpath = os.path.join(path1.parent,item[:-4])
                                out_video = os.path.join(path1.parent,item[:-4],item[:-4]+".avi")
                        else:
                            logging.info("The video "+str(item)+" had been already processed")
                            if os.path.isdir(temp_dir):
                                shutil.rmtree(temp_dir)
                            continue
                    else:
                        if not os.path.exists(os.path.join(opt.out_results,item[:-4])):
                            logging.info("Creating a new directory to store the data from "+str(item))
                            os.makedirs(os.path.join(opt.out_results,item[:-4]))
                            outpath = os.path.join(opt.out_results,item[:-4])
                            out_video = os.path.join(opt.out_results,item[:-4],item[:-4]+".avi")
                            n_frames_processed = 0
                        elif os.path.exists(os.path.join(path1.parent,item[:-4],"Cropped")):
                            if os.path.isdir(temp_dir):
                                n_frames_processed = len(os.listdir(temp_dir))
                                outpath = os.path.join(path1.parent,item[:-4])
                                out_video = os.path.join(path1.parent,item[:-4],item[:-4]+".avi")
                            else:
                                n_frames_processed = 0
                                outpath = os.path.join(path1.parent,item[:-4])
                                out_video = os.path.join(path1.parent,item[:-4],item[:-4]+".avi")
                        else:
                            logging.info("The video "+str(item)+" had been already processed")
                            if os.path.isdir(temp_dir):
                                shutil.rmtree(temp_dir)
                            continue
                    
                    # we need to convert the video into .avi
                    logging.info("Generating the .avi video")
                    #turning_video_oneimage(input_vid=svo_video,output_video=out_video)
                    if not os.path.isfile(out_video):
                        try:
                            new_to_avi(input_vid=svo_video,output_video=out_video,vid_resolution=opt.resolution)
                        except:
                            shutil.rmtree(os.path.join(path1.parent,item[:-4]))
                            continue

                    # we need to process the deepsort info
                    os.makedirs(os.path.join(outpath,"Tracked_Detections"))
                    logging.info("Generating the tracked detections from DeepSort")
                    # we need to get the deepsort config that works for the amount of fps recorded
                    if opt.resolution == "HD":
                        detect_and_track(yolo_model=opt.weights,deep_sort_model='./weights/deepsort/osnet_x1_0_imagenet.pth',config_deepsort='./weights/deepsort/deep_sort.yaml',imgsz=[opt.img_size],out=os.path.join(outpath,"Tracked_Detections"),source=out_video,conf_thres=opt.conf_thres-0.05,iou_thres=0.45,classes=[0,2])
                        deepsort_dict=reordering_deepsort_corrected(outfile=os.path.join(outpath,"Tracked_Detections","Tracked.txt"),max_separation=2)
                    else:
                        detect_and_track(yolo_model=opt.weights,deep_sort_model='./weights/deepsort/osnet_x1_0_imagenet.pth',config_deepsort='./weights/deepsort/deep_sort_15fps.yaml',imgsz=[opt.img_size],out=os.path.join(outpath,"Tracked_Detections"),source=out_video,conf_thres=opt.conf_thres-0.05,iou_thres=0.45,classes=[0,2])
                        deepsort_dict=reordering_deepsort_corrected(outfile=os.path.join(outpath,"Tracked_Detections","Tracked.txt"),max_separation=3)
                    
                    shutil.rmtree(os.path.join(outpath,"Tracked_Detections"))
                    
                    # generate results on this video
                    with torch.no_grad():
                        # all results are saved into detection_finals
                        detections_final = main()
                    
                    # creating a folder where images with and without detections will be output 
                    logging.info("Generating images with bounding boxes...")
                    os.makedirs(os.path.join(outpath,"Detections"))
                    os.makedirs(os.path.join(outpath,"No_bbox"))
                    for key in detections_final:
                        try:
                            index_retrieved = detections_final[key]["score"].index(max(detections_final[key]["score"]))
                        except:
                            index_retrieved = max(random.choice(range(len(detections_final[key]["frame"])-1)),0)
                        image_selected = detections_final[key]["frame"][index_retrieved]
                        # export the corresponding image
                        unique_identifier = uuid.uuid4().hex[:6].upper()
                        estimated_latitude = str(last_estimations(latitude_list=detections_final[key]["detection_latitude"],longitude_list=detections_final[key]["detection_longitude"],k=5)[0])
                        estimated_longitude = str(last_estimations(latitude_list=detections_final[key]["detection_latitude"],longitude_list=detections_final[key]["detection_longitude"],k=5)[1])
                        avg_score = str(max(detections_final[key]["score"]))
                        img_name = str(item[:-4])+"_"+"detection_"+str(key)+"_("+estimated_latitude+","+estimated_longitude+")_"+avg_score+".png"
                        # img_name = unique_identifier+"_"+str(item[:-4])+"_detection_"+str(key)+"_("+estimated_latitude+","+estimated_longitude+")_"+avg_score+".png"
                        # draw the bounding box
                        bbox = bounding_boxes(input_box=detections_final[key]["b_box"][index_retrieved])
                        drawing_bbox(img=image_selected,bbox=bbox,outimg=os.path.join(outpath,"Detections",img_name))
                        # copying image without bbox
                        shutil.copy(image_selected,os.path.join(outpath,"No_bbox",img_name))
                        # append this info in the dictionary
                        detections_final[key]["image_used"] = os.path.join(outpath,"Detections",img_name)
                    
                    # removing scenes
                    logging.info("Removing previous scenes...")
                    shutil.rmtree(temp_dir)

                    # Performing statistics per detection
                    logging.info("Performing the statistics for all the detections...")
                    ash_detections = dict()
                    # instantiating the dieback models
                    model2_clases = keras.models.load_model(opt.weights_class[0])
                    model4_clases = keras.models.load_model(opt.weights_class[1])
                    # looping over the detection dictionary
                    for key in detections_final:

                        if opt.resolution == "HD":
                            if (len(os.listdir(os.path.join(outpath,"Cropped",str(key)))) < 5): # we are asking for consistency - the tracker is able to track, we need consistency from the model, which provides more likelihood of being ash
                                continue
                            else:
                                health_assessment = os.path.join(outpath,"Cropped",str(key))
                                try:
                                    # Dieback = health_class_Resnet50(model2classes_path=opt.weights_class[0],model4classes_path=opt.weights_class[1],data_folder=health_assessment)
                                    Dieback = dieback_batch_class(model2_clases=model2_clases,model4_clases=model4_clases,data_folder=health_assessment)
                                    
                                except:
                                    try:
                                        # Dieback = health_class_Resnet50(model2classes_path=opt.weights_class[0],model4classes_path=opt.weights_class[1],data_folder=health_assessment)
                                        Dieback = dieback_batch_class(model2_clases=model2_clases,model4_clases=model4_clases,data_folder=health_assessment)
                                    except:
                                        print("Not working the inference")

                                if opt.rear_videos:
                                    ash_detections[key] = {
                                        "detection_id":key, # id detection
                                        "Latitude":last_rear_estimations_no_outliers(distance=detections_final[key]["distance"],latitude_list=detections_final[key]["detection_latitude"],longitude_list=detections_final[key]["detection_longitude"],k=5,avg_dist=int(opt.dist_thres))[0],
                                        "Longitude":last_rear_estimations_no_outliers(distance=detections_final[key]["distance"],latitude_list=detections_final[key]["detection_latitude"],longitude_list=detections_final[key]["detection_longitude"],k=5,avg_dist=int(opt.dist_thres))[1],                                                                                                                    
                                        "Dieback":int(round(Average_list(Dieback),0)),
                                        "Image": detections_final[key]["image_used"],
                                        "Height": int(round(np.percentile(np.array(detections_final[key]["height"]),95),0)),
                                        "Abs_height": int(round(np.percentile(np.array(detections_final[key]["abs_height"]),95),0)),
                                        "Confidence":Average_list(detections_final[key]["score"]),
                                    }  
                                else:
                                    ash_detections[key] = {
                                        "detection_id":key, # id detection
                                        "Latitude":last_estimations_no_outliers(distance=detections_final[key]["distance"],latitude_list=detections_final[key]["detection_latitude"],longitude_list=detections_final[key]["detection_longitude"],k=5,avg_dist=int(opt.dist_thres))[0],
                                        "Longitude":last_estimations_no_outliers(distance=detections_final[key]["distance"],latitude_list=detections_final[key]["detection_latitude"],longitude_list=detections_final[key]["detection_longitude"],k=5,avg_dist=int(opt.dist_thres))[1],                                                                                                                    
                                        "Dieback":int(round(Average_list(Dieback),0)),
                                        "Image": detections_final[key]["image_used"],
                                        "Height": int(round(np.percentile(np.array(detections_final[key]["height"]),95),0)),
                                        "Abs_height": int(round(np.percentile(np.array(detections_final[key]["abs_height"]),95),0)),
                                        "Confidence":Average_list(detections_final[key]["score"]),
                                    }    
                        else:
                            if (len(os.listdir(os.path.join(outpath,"Cropped",str(key)))) < 3): # we are asking for consistency - the tracker is able to track, we need consistency from the model, which provides more likelihood of being ash
                                continue
                            else:
                                health_assessment = os.path.join(outpath,"Cropped",str(key))
                                try:
                                    # Dieback = health_class_Resnet50(model2classes_path=opt.weights_class[0],model4classes_path=opt.weights_class[1],data_folder=health_assessment)
                                    Dieback = dieback_batch_class(model2_clases=model2_clases,model4_clases=model4_clases,data_folder=health_assessment)
                                except:
                                    try:
                                        # Dieback = health_class_Resnet50(model2classes_path=opt.weights_class[0],model4classes_path=opt.weights_class[1],data_folder=health_assessment)
                                        Dieback = dieback_batch_class(model2_clases=model2_clases,model4_clases=model4_clases,data_folder=health_assessment)
                                    except:
                                        print("Not working the inference")

                                if opt.rear_videos:
                                    ash_detections[key] = {
                                        "detection_id":key, # id detection
                                        "Latitude":last_rear_estimations_no_outliers(distance=detections_final[key]["distance"],latitude_list=detections_final[key]["detection_latitude"],longitude_list=detections_final[key]["detection_longitude"],k=3,avg_dist=int(opt.dist_thres))[0],
                                        "Longitude":last_rear_estimations_no_outliers(distance=detections_final[key]["distance"],latitude_list=detections_final[key]["detection_latitude"],longitude_list=detections_final[key]["detection_longitude"],k=3,avg_dist=int(opt.dist_thres))[1],                                                                                                                     
                                        "Dieback":int(round(Average_list(Dieback),0)),
                                        "Image": detections_final[key]["image_used"],
                                        "Height": int(round(np.percentile(np.array(detections_final[key]["height"]),95),0)),
                                        "Abs_height": int(round(np.percentile(np.array(detections_final[key]["abs_height"]),95),0)),
                                        "Confidence":Average_list(detections_final[key]["score"])
                                    }
                                else:
                                    ash_detections[key] = {
                                        "detection_id":key, # id detection
                                        "Latitude":last_estimations_no_outliers(distance=detections_final[key]["distance"],latitude_list=detections_final[key]["detection_latitude"],longitude_list=detections_final[key]["detection_longitude"],k=3,avg_dist=int(opt.dist_thres))[0],
                                        "Longitude":last_estimations_no_outliers(distance=detections_final[key]["distance"],latitude_list=detections_final[key]["detection_latitude"],longitude_list=detections_final[key]["detection_longitude"],k=3,avg_dist=int(opt.dist_thres))[1],                                                                                                                     
                                        "Dieback":int(round(Average_list(Dieback),0)),
                                        "Image": detections_final[key]["image_used"],
                                        "Height": int(round(np.percentile(np.array(detections_final[key]["height"]),95),0)),
                                        "Abs_height": int(round(np.percentile(np.array(detections_final[key]["abs_height"]),95),0)),
                                        "Confidence":Average_list(detections_final[key]["score"])
                                    }
                    # removing images
                    shutil.rmtree(os.path.join(outpath,"Cropped"))
                    # generating shapefile
                    logging.info("Generating the shapefiles...")
                    os.makedirs(os.path.join(outpath,"Shapefiles"))
                    shp_name = item[:-4]+"_det.shp"
                    shapefile_from_dict(input_dict=ash_detections,out_directory=os.path.join(outpath,"Shapefiles"),file_name=shp_name)
                    logging.info("Shapefiles have been generated successfully")
                    # releasing cuda
                    # cuda.select_device(0)
                    # cuda.close()
                    torch.cuda.empty_cache()
                
                except Exception as e: 
                    # logging the error
                    print(e)
                    exception_type, exception_object, exception_traceback = sys.exc_info()
                    filename = exception_traceback.tb_frame.f_code.co_filename
                    line_number = exception_traceback.tb_lineno

                    print("Exception type: ", exception_type)
                    print("File name: ", filename)
                    print("Line number: ", line_number)

                    logging.info("Error while processing video "+str(item))
                    # sending emails with the error
                    subject = "Error while processing video"
                    content = "Error while processing video "+str(item)
            
            elif item.endswith(".svo") and (item.split("_")[1]=="front") and (opt.rear_videos == False):
                try:
                    logging.info("dealing with video "+str(item))
                    # we generate the absolute path of the video
                    svo_video = os.path.join(opt.svo,item)
                    # we need to make a directory for the output, so we can save the video and the images
                    if opt.out_results is None:
                        path1 = Path(svo_video)
                        if not os.path.exists(os.path.join(path1.parent,item[:-4])):
                            logging.info("Creating a new directory to store the data from "+str(item))
                            os.makedirs(os.path.join(path1.parent,item[:-4]))
                            outpath = os.path.join(path1.parent,item[:-4])
                            out_video = os.path.join(path1.parent,item[:-4],item[:-4]+".avi")
                            n_frames_processed = 0
                        elif os.path.exists(os.path.join(path1.parent,item[:-4],"Cropped")):
                            if os.path.isdir(temp_dir):
                                n_frames_processed = len(os.listdir(temp_dir))
                                outpath = os.path.join(path1.parent,item[:-4])
                                out_video = os.path.join(path1.parent,item[:-4],item[:-4]+".avi")
                            else:
                                n_frames_processed = 0
                                outpath = os.path.join(path1.parent,item[:-4])
                                out_video = os.path.join(path1.parent,item[:-4],item[:-4]+".avi")
                        else:
                            logging.info("The video "+str(item)+" had been already processed")
                            if os.path.isdir(temp_dir):
                                shutil.rmtree(temp_dir)
                            continue
                    else:
                        if not os.path.exists(os.path.join(opt.out_results,item[:-4])):
                            logging.info("Creating a new directory to store the data from "+str(item))
                            os.makedirs(os.path.join(opt.out_results,item[:-4]))
                            outpath = os.path.join(opt.out_results,item[:-4])
                            out_video = os.path.join(opt.out_results,item[:-4],item[:-4]+".avi")
                            n_frames_processed = 0
                        elif os.path.exists(os.path.join(path1.parent,item[:-4],"Cropped")):
                            if os.path.isdir(temp_dir):
                                n_frames_processed = len(os.listdir(temp_dir))
                                outpath = os.path.join(path1.parent,item[:-4])
                                out_video = os.path.join(path1.parent,item[:-4],item[:-4]+".avi")
                            else:
                                n_frames_processed = 0
                                outpath = os.path.join(path1.parent,item[:-4])
                                out_video = os.path.join(path1.parent,item[:-4],item[:-4]+".avi")
                        else:
                            logging.info("The video "+str(item)+" had been already processed")
                            if os.path.isdir(temp_dir):
                                shutil.rmtree(temp_dir)
                            continue
                    
                    # we need to convert the video into .avi
                    logging.info("Generating the .avi video")
                    #turning_video_oneimage(input_vid=svo_video,output_video=out_video)
                    if not os.path.isfile(out_video):
                        new_to_avi(input_vid=svo_video,output_video=out_video,vid_resolution=opt.resolution)
                    
                    # we need to process the deepsort info
                    os.makedirs(os.path.join(outpath,"Tracked_Detections"))
                    logging.info("Generating the tracked detections from DeepSort")
                    # we need to get the deepsort config that works for the amount of fps recorded
                    if opt.resolution == "HD":
                        detect_and_track(yolo_model=opt.weights,deep_sort_model='./weights/deepsort/osnet_x1_0_imagenet.pth',config_deepsort='./weights/deepsort/deep_sort.yaml',imgsz=[opt.img_size],out=os.path.join(outpath,"Tracked_Detections"),source=out_video,conf_thres=opt.conf_thres-0.05,iou_thres=0.45,classes=[0,2])
                        deepsort_dict=reordering_deepsort_corrected(outfile=os.path.join(outpath,"Tracked_Detections","Tracked.txt"),max_separation=2)
                    else:
                        detect_and_track(yolo_model=opt.weights,deep_sort_model='./weights/deepsort/osnet_x1_0_imagenet.pth',config_deepsort='./weights/deepsort/deep_sort_15fps.yaml',imgsz=[opt.img_size],out=os.path.join(outpath,"Tracked_Detections"),source=out_video,conf_thres=opt.conf_thres-0.05,iou_thres=0.45,classes=[0,2])
                        deepsort_dict=reordering_deepsort_corrected(outfile=os.path.join(outpath,"Tracked_Detections","Tracked.txt"),max_separation=3)
                    
                    shutil.rmtree(os.path.join(outpath,"Tracked_Detections"))
                    
                    # generate results on this video
                    with torch.no_grad():
                        # all results are saved into detection_finals
                        detections_final = main()
                    
                    # creating a folder where images with and without detections will be output 
                    logging.info("Generating images with bounding boxes...")
                    os.makedirs(os.path.join(outpath,"Detections"))
                    os.makedirs(os.path.join(outpath,"No_bbox"))
                    for key in detections_final:
                        try:
                            index_retrieved = detections_final[key]["score"].index(max(detections_final[key]["score"]))
                        except:
                            index_retrieved = max(random.choice(range(len(detections_final[key]["frame"])-1)),0)
                        image_selected = detections_final[key]["frame"][index_retrieved]
                        # export the corresponding image
                        unique_identifier = uuid.uuid4().hex[:6].upper()
                        estimated_latitude = str(last_estimations(latitude_list=detections_final[key]["detection_latitude"],longitude_list=detections_final[key]["detection_longitude"],k=5)[0])
                        estimated_longitude = str(last_estimations(latitude_list=detections_final[key]["detection_latitude"],longitude_list=detections_final[key]["detection_longitude"],k=5)[1])
                        avg_score = str(max(detections_final[key]["score"]))
                        img_name = str(item[:-4])+"_"+"detection_"+str(key)+"_("+estimated_latitude+","+estimated_longitude+")_"+avg_score+".png"
                        # img_name = unique_identifier+"_"+str(item[:-4])+"_detection_"+str(key)+"_("+estimated_latitude+","+estimated_longitude+")_"+avg_score+".png"
                        # draw the bounding box
                        bbox = bounding_boxes(input_box=detections_final[key]["b_box"][index_retrieved])
                        drawing_bbox(img=image_selected,bbox=bbox,outimg=os.path.join(outpath,"Detections",img_name))
                        # copying image without bbox
                        shutil.copy(image_selected,os.path.join(outpath,"No_bbox",img_name))
                        # append this info in the dictionary
                        detections_final[key]["image_used"] = os.path.join(outpath,"Detections",img_name)
                    
                    # removing scenes
                    logging.info("Removing previous scenes...")
                    shutil.rmtree(temp_dir)

                    # Performing statistics per detection
                    logging.info("Performing the statistics for all the detections...")
                    ash_detections = dict()
                    # instantiating models
                    model2_clases = keras.models.load_model(opt.weights_class[0])
                    model4_clases = keras.models.load_model(opt.weights_class[1])
                    # looping over detections
                    for key in detections_final:

                        if opt.resolution == "HD":
                            if (len(os.listdir(os.path.join(outpath,"Cropped",str(key)))) < 5): # we are asking for consistency - the tracker is able to track, we need consistency from the model, which provides more likelihood of being ash
                                continue
                            else:
                                health_assessment = os.path.join(outpath,"Cropped",str(key))
                                try:
                                    # Dieback = health_class_Resnet50(model2classes_path=opt.weights_class[0],model4classes_path=opt.weights_class[1],data_folder=health_assessment)
                                    Dieback = dieback_batch_class(model2_clases=model2_clases,model4_clases=model4_clases,data_folder=health_assessment)
                                except:
                                    try:
                                        # Dieback = health_class_Resnet50(model2classes_path=opt.weights_class[0],model4classes_path=opt.weights_class[1],data_folder=health_assessment)
                                        Dieback = dieback_batch_class(model2_clases=model2_clases,model4_clases=model4_clases,data_folder=health_assessment)
                                    except:
                                        print("Not working the inference")

                                ash_detections[key] = {
                                    "detection_id":key, # id detection
                                    "Latitude":last_estimations_no_outliers(distance=detections_final[key]["distance"],latitude_list=detections_final[key]["detection_latitude"],longitude_list=detections_final[key]["detection_longitude"],k=5,avg_dist=int(opt.dist_thres))[0],
                                    "Longitude":last_estimations_no_outliers(distance=detections_final[key]["distance"],latitude_list=detections_final[key]["detection_latitude"],longitude_list=detections_final[key]["detection_longitude"],k=5,avg_dist=int(opt.dist_thres))[1],                                                                                                                    
                                    "Dieback":int(round(Average_list(Dieback),0)),
                                    "Image": detections_final[key]["image_used"],
                                    "Height": int(round(np.percentile(np.array(detections_final[key]["height"]),95),0)),
                                    "Abs_height": int(round(np.percentile(np.array(detections_final[key]["abs_height"]),95),0)),
                                    "Confidence":Average_list(detections_final[key]["score"]),
                                }    
                        else:
                            if (len(os.listdir(os.path.join(outpath,"Cropped",str(key)))) < 3): # we are asking for consistency - the tracker is able to track, we need consistency from the model, which provides more likelihood of being ash
                                continue
                            else:
                                health_assessment = os.path.join(outpath,"Cropped",str(key))
                                try:
                                    # Dieback = health_class_Resnet50(model2classes_path=opt.weights_class[0],model4classes_path=opt.weights_class[1],data_folder=health_assessment)
                                    Dieback = dieback_batch_class(model2_clases=model2_clases,model4_clases=model4_clases,data_folder=health_assessment)
                                except:
                                    try:
                                        # Dieback = health_class_Resnet50(model2classes_path=opt.weights_class[0],model4classes_path=opt.weights_class[1],data_folder=health_assessment)
                                        Dieback = dieback_batch_class(model2_clases=model2_clases,model4_clases=model4_clases,data_folder=health_assessment)
                                    except:
                                        print("Not working the inference")

                                ash_detections[key] = {
                                    "detection_id":key, # id detection
                                    "Latitude":last_estimations_no_outliers(distance=detections_final[key]["distance"],latitude_list=detections_final[key]["detection_latitude"],longitude_list=detections_final[key]["detection_longitude"],k=3,avg_dist=int(opt.dist_thres))[0],
                                    "Longitude":last_estimations_no_outliers(distance=detections_final[key]["distance"],latitude_list=detections_final[key]["detection_latitude"],longitude_list=detections_final[key]["detection_longitude"],k=3,avg_dist=int(opt.dist_thres))[1],                                                                                                                     
                                    "Dieback":int(round(Average_list(Dieback),0)),
                                    "Image": detections_final[key]["image_used"],
                                    "Height": int(round(np.percentile(np.array(detections_final[key]["height"]),95),0)),
                                    "Abs_height": int(round(np.percentile(np.array(detections_final[key]["abs_height"]),95),0)),
                                    "Confidence":Average_list(detections_final[key]["score"])
                                }
                    # removing images
                    shutil.rmtree(os.path.join(outpath,"Cropped"))
                    # generating shapefile
                    logging.info("Generating the shapefiles...")
                    os.makedirs(os.path.join(outpath,"Shapefiles"))
                    shp_name = item[:-4]+"_det.shp"
                    shapefile_from_dict(input_dict=ash_detections,out_directory=os.path.join(outpath,"Shapefiles"),file_name=shp_name)
                    logging.info("Shapefiles have been generated successfully")
                    # releasing cuda
                    # cuda.select_device(0)
                    # cuda.close()
                    torch.cuda.empty_cache()
                
                except Exception as e: 
                    # logging the error
                    print(e)
                    exception_type, exception_object, exception_traceback = sys.exc_info()
                    filename = exception_traceback.tb_frame.f_code.co_filename
                    line_number = exception_traceback.tb_lineno

                    print("Exception type: ", exception_type)
                    print("File name: ", filename)
                    print("Line number: ", line_number)

                    logging.info("Error while processing video "+str(item))
                    # sending emails with the error
                    subject = "Error while processing video"
                    content = "Error while processing video "+str(item)

        end = time.time()
        elapsed = end - start
        print("Time elapsed: "+str(elapsed))

        # merging data
        # ============
        # subfolder names
        print("Listing directories")

        all_items = os.listdir(opt.svo)
        all_subfolders = []
        for file_ in all_items:
            if file_.endswith(".svo"):
                all_subfolders.append(file_[:-4])

        print("Svo videos appended")
        
        # merging all the shapefiles
        print("Proceeding to merging shp")
        os.makedirs(os.path.join(opt.svo,"Shapefile"))
        proceed = merging_shp(input_dir=opt.svo,sub_folders=all_subfolders)
        print("Shp merged")
        if proceed:
            # creating a unified folder with all detection images
            os.makedirs(os.path.join(opt.svo,"Detections"))
            for sub in all_subfolders:
                detection_folder = os.path.join(opt.svo,sub,"Detections")
                shutil.copytree(detection_folder, os.path.join(opt.svo,"Detections"),dirs_exist_ok=True)
            
            # creating a unified fodler with no-bbox images 
            os.makedirs(os.path.join(opt.svo,"No_bbox"))

            for sub in all_subfolders:
                detection_folder = os.path.join(opt.svo,sub,"No_bbox")
                shutil.copytree(detection_folder, os.path.join(opt.svo,"No_bbox"),dirs_exist_ok=True)
        
            # Copy .avi files
            print("Proceeding to removing videos")
            video_files = []
            for root,dirs,files in os.walk(opt.svo):
                for fle in files:
                    if fle.endswith(".avi"):
                        video_files.append(os.path.join(root,fle))
            
            for item in video_files:
                os.remove(item)

            print("All videos removed")
            # copy all the monitoring
            os.makedirs(os.path.join(opt.svo,"Monitoring"))
            os.makedirs(os.path.join(opt.svo,"Monitoring","Images"))
            os.makedirs(os.path.join(opt.svo,"Monitoring","Labels"))
            print("copying monitoring images...")
            for sub in all_subfolders:
                mon_im_folder = os.path.join(opt.svo,sub,"Monitoring","Images")
                shutil.copytree(mon_im_folder, os.path.join(opt.svo,"Monitoring","Images"),dirs_exist_ok=True)
                mon_lbl_folder = os.path.join(opt.svo,sub,"Monitoring","Labels")
                shutil.copytree(mon_lbl_folder, os.path.join(opt.svo,"Monitoring","Labels"),dirs_exist_ok=True)
            print("monitoring images copied")
            # Removing subfolders
            for fldr in all_subfolders:
                shutil.rmtree(os.path.join(opt.svo,fldr))
            print("Removed subfolders")
            # Removing svo files
            svo_files = []
            for vdd in all_items:
                if vdd.endswith(".svo"):
                   svo_files.append(os.path.join(opt.svo,vdd)) 
            
            for vdd in svo_files:
                os.remove(vdd)
            print("Removed Video files")
            # Removing mt data
            txt_files = []
            for vdd in all_items:
                if vdd.endswith(".txt"):
                   txt_files.append(os.path.join(opt.svo,vdd)) 
            
            for vdd in txt_files:
                os.remove(vdd)
            print("Removed MT data")
            with open(os.path.join(opt.svo,'state.data'), 'w') as f:
                f.write('ProcessFinished')
            # # blurring faces 
            # blur_faces(input_folder=os.path.join(opt.svo,"No_bbox"),model='./weights/blur_faces/face.pb',threshold=0.15)
            # blur_faces(input_folder=os.path.join(opt.svo,"Detections"),model='./weights/blur_faces/face.pb',threshold=0.15)
            # print("Faces have been blurred")
            # blur plate numbers
            # from detect_lp import blur_license_plates
            # blur_license_plates(output=os.path.join(opt.svo,"No_bbox"),source=os.path.join(opt.svo,"No_bbox"),weights="./weights/blur_plates/detection_best.pt",img_size=1024,view_img=False,save_txt=False,conf_thres=0.25,update=False)
            # backing up raw shapefile
            # os.makedirs(os.path.join(opt.svo,"Shapefile_raw"))
            # shutil.copytree(os.path.join(opt.svo,"Shapefile"),os.path.join(opt.svo,"Shapefile_raw"),dirs_exist_ok=True)
            # # post processing
            # auto_processing(first_time=True, detection_shp=os.path.join(opt.svo,"Shapefile","merge.shp"),ntm_shp=r"\\gb010587mm\IDA_datasets\export_shapefile\OneDrive_1_30-11-2022\merge.shp",road_shp=r"\\gb010587mm\IDA_datasets\export_shapefile\OneDrive_2_30-11-2022\Ceredigion_dissolved.shp")

    else:
        # Initiating logging
        # ==================
        logging.basicConfig(filename=os.path.join(opt.svo,"logs.log"), level=logging.DEBUG)

        with torch.no_grad():
            main()
        path1 = Path(opt.svo)
        os.makedirs(os.path.join(path1.parent,"results"))
        out_video = os.path.join(path1.parent,"results","translated.avi")
        turning_video(input_vid=opt.svo,output_video=out_video)