# Package information
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
import cv2
import logging
import numpy as np
import os
from pathlib import Path
import pyzed.sl as sl
import torch.backends.cudnn as cudnn
import shutil
from threading import Lock, Thread, Event
import time
from time import sleep
import torch
import random
import sys
import uuid

# importing Zed dependencies
# ===========================
sys.path.insert(1, './src/pipeline')
import cv_viewer.tracking_viewer as cv_viewer
from image_preprocessing import img_preprocess,xywh2abcd,bounding_boxes,drawing_bbox,drawing_bbox_and_distance,detections_to_custom_box
import ogl_viewer.viewer as gl
from video_generation import progress_bar, turning_video_oneimage

# importing Yolo dependencies
# ===========================
sys.path.insert(0, './models/Object_detection/yolov5')
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.torch_utils import select_device
from utils.augmentations import letterbox

# Initiating Threading
lock = Lock()
run_signal = False
exit_signal = False

# Adding extra functionality
# =========================
def torch_thread(weights, img_size, conf_thres=0.2, iou_thres=0.45):
    global image_net, exit_signal, run_signal, detections

    logging.info("Intializing Network...")

    device = select_device()
    half = device.type != 'cpu'  # half precision only supported on CUDA
    imgsz = img_size

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16
    cudnn.benchmark = True

    # Run inference
    if (device.type != 'cpu'):
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    
    while not exit_signal:
        if run_signal:
            lock.acquire()
            img, ratio, pad = img_preprocess(image_net, device, half, imgsz)

            pred = model(img)[0]
            det = non_max_suppression(pred, conf_thres, iou_thres)

            # ZED CustomBox format (with inverse letterboxing tf applied)
            detections = detections_to_custom_box(det, img, image_net)
            lock.release()
            run_signal = False
        sleep(0.01)
        

def main():
    # First, we are going to define some global variables
    global image_net, exit_signal, run_signal, detections, temp_dir
    
    # Defining the thread
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
    To configure the depth sensing, we use InitPArameters at initialization and RuntimeParameters to change specific parameters during use.
    Basically,
    initParam - Holds the options used to initialised the Camera object
    runtimeParam - Parameters that defines the behaviour of the grab.
    '''
    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
    init_params.camera_resolution = sl.RESOLUTION.HD1080 # Selected a 1920x1080 resolution
    init_params.camera_fps = 30 # camera set as 30 fps
    init_params.coordinate_units = sl.UNIT.METER # measurements in meters
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Ultra mode (offers the highest depth range and better preserves Z-accuracy along the sensing range)
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_maximum_distance = 50 # set a maximum depth perception distance to 1m
    init_params.depth_minimum_distance = 1 # set a minimum depth perception distance to 1m

    runtime_params = sl.RuntimeParameters()
    runtime_params.sensing_mode = sl.SENSING_MODE.STANDARD # preserves edges and depth accuracy.
    runtime_params.confidence_threshold = 50 # Threshold to reject depth values based on their confidence
    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        logging.warning(repr(status))
        exit()

    image_left_tmp = sl.Mat()

    logging.info("Initialized Camera")

    # positional tracking and detection parameters
    '''
    Positional parameters - These are parameters for tracking initialisation
    ObjectDetection - object detection parameters
    Objects - Contains the result of the object detection module
    '''
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    positional_tracking_parameters.set_floor_as_origin = False
    zed.enable_positional_tracking(positional_tracking_parameters)

    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.DETECTION_MODEL.CUSTOM_BOX_OBJECTS # sets as custom_box_objects as we have defined a function for this
    obj_param.enable_tracking = True
    zed.enable_object_detection(obj_param)

    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    obj_runtime_param.detection_confidence_threshold = np.float16(opt.conf_thres) # detection confidence threshold
    obj_runtime_param.object_class_filter = {} # select which object types to detect and track

    # Display options

    camera_infos = zed.get_camera_information()
    # Create OpenGL viewer
    viewer = gl.GLViewer()
    point_cloud_res = sl.Resolution(min(camera_infos.camera_resolution.width, 720),
                                    min(camera_infos.camera_resolution.height, 404))
    point_cloud_render = sl.Mat()
    viewer.init(camera_infos.camera_model, point_cloud_res, obj_param.enable_tracking)
    point_cloud = sl.Mat(point_cloud_res.width, point_cloud_res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    image_left = sl.Mat()
    
    # Utilities for 2D display
    display_resolution = sl.Resolution(min(camera_infos.camera_resolution.width, 1280),
                                       min(camera_infos.camera_resolution.height, 720))
    image_scale = [display_resolution.width / camera_infos.camera_resolution.width, display_resolution.height / camera_infos.camera_resolution.height]
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
        
        temp_dir = os.path.join(opt.svo,"scenes")
        os.makedirs(temp_dir)
    else:
        path = Path(opt.svo)
        
        temp_dir = path.parent.absolute()
        os.makedirs(temp_dir)

    # Finally, we loop through all the scenes
    detections_dict = dict()
    frame_number = -1
    while viewer.is_available() and not exit_signal:
        frame_number = frame_number + 1
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # -- Get the image/scene
            lock.acquire()
            zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
            image_net = image_left_tmp.get_data()
            # -- save scene
           
            # Convert SVO image from RGBA to RGB
            ocv_image_sbs_rgb = cv2.cvtColor(image_net, cv2.COLOR_RGBA2RGB)

            # Write the RGB image in the video
            cv2.imwrite(os.path.join(temp_dir,str(frame_number)+".jpg"),ocv_image_sbs_rgb)
            lock.release()
            run_signal = True

            # -- Detection running on the other thread
            while run_signal:
                sleep(0.001)

            # Wait for detections
            lock.acquire()
            # -- Ingest detections
            zed.ingest_custom_box_objects(detections)
            lock.release()
            zed.retrieve_objects(objects, obj_runtime_param)

            # -- Display
            # Retrieve display data
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, point_cloud_res)
            point_cloud.copy_to(point_cloud_render)
            zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)

            # 3D rendering
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
            for object in objects.object_list:
                '''
                The Objects class stores all the information regarding the different objects present in the scene in it object_list attribute. 
                Each individual object is stored as a ObjectData with all information about it, such as bounding box, position, mask, etc.
                All objects from a given frame are stored in a vector within Objects. Objects also contains the timestamp of the detection, 
                which can help connect the objects to the images 
                '''
                # first, we filter out those detections that are above the confidence threshold and closer than 20 m
                if (((object.confidence > opt.conf_thres) and (object.position[2] > -50))):
                    
                    # we append information to the corresponding key(id)
                    if (object.id in detections_dict):
                        
                        detections_dict[object.id]["distance"].append(object.position[2])
                        detections_dict[object.id]["frame"].append(os.path.join(temp_dir,str(frame_number)+".jpg"))
                        detections_dict[object.id]["b_box"].append(object.bounding_box_2d)
                        detections_dict[object.id]["timestamp"].append(objects.timestamp.get_nanoseconds())
                            
                    else:
                                
                        detections_dict[object.id]={"distance":[object.position[2]],"frame":[os.path.join(temp_dir,str(frame_number)+".jpg")],"b_box":[object.bounding_box_2d],"timestamp":[objects.timestamp.get_nanoseconds()]}
                       
        else:
            exit_signal = True
        
    viewer.exit()
    exit_signal = True
    zed.close()
    
    return(detections_dict)


# The script
# ===========
if __name__ == '__main__':

    # Defining the parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default="./data", help='optional svo file or folder')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--out_results', type=str, default="./data/outputs", help='Output where results will be generated')
    opt = parser.parse_args()

    # Adding flexibility - we can provide either a folder with videos or a video
    if os.path.isdir(opt.svo):
        if os.path.isdir(os.path.join(opt.svo,"scenes")):
            shutil.rmtree(os.path.join(opt.svo,"scenes"))

        # Initiating logging
        # ==================
        filename_logs = os.path.join(r"\\gb010587mm\Software_dev\Ash_Dieback_Solution\fast","runtime.txt")
        logging.basicConfig(filename=filename_logs, level=logging.INFO, filemode="a",
            format='%(levelname)s:%(message)s')

        logging.info("I have been provided a folder with videos")
        for item in os.listdir(opt.svo):

            

            if item.endswith(".svo"):
                
                logging.info("dealing with video "+str(item))
                # we generate the absolute path of the video
                svo_video = os.path.join(opt.svo,item)
                # generate results on this video
                with torch.no_grad():
                    # all results are saved into detection_finals
                    detections_final = main()

                # we need to make a directory for the output, so we can save the video and the images
                if opt.out_results is None:
                    path1 = Path(svo_video)
                    if not os.path.exists(os.path.join(path1.parent,item[:-4])):
                        logging.info("Creating a new directory to store the data from "+str(item))
                        os.makedirs(os.path.join(path1.parent,item[:-4]))
                        outpath = os.path.join(path1.parent,item[:-4])
                        out_video = os.path.join(path1.parent,item[:-4],item[:-4]+".avi")
                    else:
                        logging.info("The video "+str(item)+" had been already processed")
                        shutil.rmtree(temp_dir)
                        continue
                else:
                    if not os.path.exists(os.path.join(opt.out_results,item[:-4])):
                        logging.info("Creating a new directory to store the data from "+str(item))
                        os.makedirs(os.path.join(opt.out_results,item[:-4]))
                        outpath = os.path.join(opt.out_results,item[:-4])
                        out_video = os.path.join(opt.out_results,item[:-4],item[:-4]+".avi")
                    else:
                        logging.info("The video "+str(item)+" had been already processed")
                        shutil.rmtree(temp_dir)
                        continue

                # we convert the video into .avi
                turning_video_oneimage(input_vid=svo_video,output_video=out_video)

                # creating a folder where images with and without detections will be output
                
                logging.info("Generating images with bounding boxes...")
                os.makedirs(os.path.join(outpath,"Detections"))
                os.makedirs(os.path.join(outpath,"No_bbox"))
                for key in detections_final:
                    try:
                        index_retrieved = random.choice(range(len(detections_final[key]["frame"])-1))
                    except:
                        index_retrieved = 0
                    image_selected = detections_final[key]["frame"][index_retrieved]
                    # export the corresponding image
                    unique_identifier = uuid.uuid4().hex[:6].upper()
                    img_name = unique_identifier+"_"+str(item[:-4])+"_detection_"+str(key)+".png"
                    # draw the bounding box
                    bbox = bounding_boxes(input_box=detections_final[key]["b_box"][index_retrieved])
                    drawing_bbox(img=image_selected,bbox=bbox,outimg=os.path.join(outpath,"Detections",img_name))
                    # copying image without bbox
                    shutil.copy(image_selected,os.path.join(outpath,"No_bbox",img_name))
                    # append this info in the dictionary
                    detections_final[key]["image_used"] = os.path.join(outpath,"Detections",img_name)
                
                # creating a folder with the cropped images
                logging.info("Generating cropped images")
                os.makedirs(os.path.join(outpath,"Cropped"))
                for key in detections_final:
                    os.makedirs(os.path.join(outpath,"Cropped",str(key)))
                    for idx,scene in enumerate(detections_final[key]["frame"]):
                        # reading image
                        original_image= cv2.imread(scene)
                        # generating the cropping box
                        cropping_box = detections_final[key]["b_box"][idx]
                        y =int(min(cropping_box[0][1],cropping_box[1][1],cropping_box[2][1],cropping_box[3][1]))
                        h = int(max(cropping_box[0][1],cropping_box[1][1],cropping_box[2][1],cropping_box[3][1]) - min(cropping_box[0][1],cropping_box[1][1],cropping_box[2][1],cropping_box[3][1]))
                        x =int(min(cropping_box[0][0],cropping_box[1][0],cropping_box[2][0],cropping_box[3][0]))
                        w = int(max(cropping_box[0][0],cropping_box[1][0],cropping_box[2][0],cropping_box[3][0]) - min(cropping_box[0][0],cropping_box[1][0],cropping_box[2][0],cropping_box[3][0]))
                        # generating the cropping image
                        cropped_image = original_image[y:y+h,x:x+w]
                        # saving the cropped image
                        try:
                            cv2.imwrite(os.path.join(outpath,"Cropped",str(key),str(idx)+".jpg"),cropped_image)
                        except:
                            print("image skipped")

                # removing scenes
                logging.info("Removing previous scenes...")
                shutil.rmtree(temp_dir)

                # evaluating dieback level

                # estimating coordinates

                # generating shapefile







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





