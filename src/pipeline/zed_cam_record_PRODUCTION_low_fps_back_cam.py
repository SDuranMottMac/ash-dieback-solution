import pyzed.sl as sl # manual setup
import cv2 # pip install opencv-python
import numpy as np # pip install numpy
from PIL import ImageStat, Image # pip install pillow
import tkinter as tk # core module
from tkinter import messagebox # core module
import math # core module
from tkinter.ttk import * # core module
import datetime # core module
import threading # core module
import sys # core module
import os # core module
import csv # core module
import time # core module
import random # core module


# WANTED FUNCTIONALITY:
## - ERROR CATCHING SYSTEM - CAMERA MATCHING - QUALITY -- TODO
## - QUALITY SAMPLING - SETTINGS CORRECTION -- TODO
## - BACKUP AND EXPORT -- TODO

# BRIGHTNESS 	Defines the brightness control. Affected value should be between 0 and 8.
# CONTRAST 	    Defines the contrast control. Affected value should be between 0 and 8.
# HUE 	        Defines the hue control. Affected value should be between 0 and 11.
# SATURATION 	Defines the saturation control. Affected value should be between 0 and 8.
# SHARPNESS 	Defines the digital sharpening control. Affected value should be between 0 and 8.
# GAMMA 	    Defines the ISP gamma control. Affected value should be between 1 and 9.
# GAIN 	        Defines the gain control. Affected value should be between 0 and 100 for manual control.
# EXPOSURE 	    Defines the exposure control. Affected value should be between 0 and 100 for manual control.
#             The exposition is mapped linearly in a percentage of the following max values. Special case for the setExposure(0) that corresponds to 0.17072ms.
# AEC_AGC 	    Defines if the Gain and Exposure are in automatic mode or not. Setting a Gain or Exposure through @GAIN or @EXPOSURE values will automatically set this value to 0.
# AEC_AGC_ROI   Defines the region of interest for automatic exposure/gain computation. To be used with overloaded @setCameraSettings/@getCameraSettings functions.
# WHITEBALANCE_TEMPERATURE 	Defines the color temperature value. Setting a value will automatically set @WHITEBALANCE_AUTO to 0. Affected value should be between 2800 and 6500 with a step of 100.
# WHITEBALANCE_AUTO 	Defines if the White balance is in automatic mode or not
# LED_STATUS 	Defines the status of the camera front LED. Set to 0 to disable the light, 1 to enable the light. Default value is on. Requires Camera FW 1523 at least.

CHECKED_FRAME_0 = 10 # we check the frequency each 10 frames
CHECKED_FRAME_1 = 15

class tkinterGUI():

    def __init__(self):
        self.chapter_no = [0,0]
        self.auto_bright = True
        self.toggle_auto_brightness = True
        self.csv_list = []
        self.run()


    def run(self):
        # creates a Tk() object
        self.root = tk.Tk()
        # define the title
        self.root.title("Zed Camera Recorder")
        # define the geometry
        self.root.geometry("300x300")
        # define the label
        label = Label(self.root, text ="Zed Camera Recorder", font=20)
        label.pack(pady=10)

        # GUI config
        Button(self.root, text="Camera Set-Up", command=self.cam_set_up, width=30).pack(pady=10)
        Button(self.root, text="Stop Recording", command=self.stop_camera, width=30).pack(pady=10)
        Button(self.root, text="Camera Settings", command=self.optimise_thread, width=30).pack(pady=10)

        quit = Button(self.root, text="QUIT", command=self.write_to_csv).pack(side="bottom")

        self.root.mainloop()


    def cam_set_up(self):
        def selections():
            frames = fps.get()
            resolution = res.get()
            if frames == 1:
                self.fps=15
            elif frames == 2:
                self.fps=30
            else:
                self.fps=15 # why else is 30 fps?
            if resolution == 1:
                self.res=sl.RESOLUTION.HD720
            elif resolution == 2:
                self.res=sl.RESOLUTION.HD1080
            elif resolution == 3:
                self.res=sl.RESOLUTION.HD2K
            else:
                self.res=sl.RESOLUTION.HD2K
            if chap_mins.get() == "":
                self.mins_p_chap = 8
            else:
                self.mins_p_chap = float(chap_mins.get())
            
            if min_exp.get() == "":
                self.min_exp = 0
            else:
                self.min_exp = int(min_exp.get())
            if max_exp.get() == "":
                self.max_exp = 100
            else:
                self.max_exp = int(max_exp.get())
            if min_gain.get() == "":
                self.min_gain = 0
            else:
                self.min_gain = int(min_gain.get())
            if max_gain.get() == "":
                self.max_gain = 100
            else:
                self.max_gain = int(max_gain.get())

            if lum_targ.get() == "":
                self.lum_targ = 155
            else:
                self.lum_targ = int(lum_targ.get())
            
            if gain_factor.get() == "":
                self.gain_factor = 2
            else:
                self.gain_factor = int(gain_factor.get())

            if lum_threshold.get() == "":
                self.lum_threshold = 50
            else:
                self.lum_threshold = int(lum_threshold.get())
            
            self.start_camera_thread()


        def toggle_exp():
            if self.toggle_auto_brightness == True:
                self.toggle_auto_brightness = False
                info = "off"
            elif self.toggle_auto_brightness == False:
                self.toggle_auto_brightness = True
                info = "on"
            messagebox.showinfo("Setting Changed", "Auto Brightness adjustment is {}".format(info))
            

        Window = tk.Toplevel(self.root)
        Window.title("Camera Configuration")
        Window.geometry("750x350")
        tk.Label(Window, width=3).grid(column=0)
        Label(Window, text="Camera Configuration", font=10).grid(row=1, column=2, columnspan=2)

        # Asking the user for the fps
        fps = tk.IntVar()
        Label(Window, text="FPS", font=8).grid(row=2, column=1)
        Radiobutton(Window, text="15 fps", variable=fps, value=1).grid(row=2, column=2)
        Radiobutton(Window, text="30 fps", variable=fps, value=2).grid(row=2, column=3)
        # Asking the user for the resolution to use
        res = tk.IntVar()
        Label(Window, text="Resolution", font=8).grid(row=3, column=1)
        Radiobutton(Window, text="720p", variable=res, value=1).grid(row=3, column=2)
        Radiobutton(Window, text="1080p", variable=res, value=2).grid(row=3, column=3)
        Radiobutton(Window, text="2k", variable=res, value=3).grid(row=3, column=4)
        # Asking the user for the number of minutes per chapter
        Label(Window, text="Minutes per Chapter", font=8).grid(row=4, column=1)
        chap_mins = Entry(Window)
        chap_mins.grid(row=4, column=2)
        # Asking the user for the luminescence target
        Label(Window, text="Luminescence Target", font=8).grid(row=4, column=3)
        lum_targ = Entry(Window)
        lum_targ.grid(row=4, column=4)
        # Asking the user for the minimum exposure
        Label(Window, text="Min Exposure", font=8).grid(row=5, column=1)
        min_exp = Entry(Window)
        min_exp.grid(row=5, column=2)
        # Asking the user for the maximum exposure
        Label(Window, text="Max Exposure", font=8).grid(row=5, column=3)
        max_exp = Entry(Window)
        max_exp.grid(row=5, column=4)
        # Asking the user for the minimum gain
        Label(Window, text="Min Gain", font=8).grid(row=6, column=1)
        min_gain = Entry(Window)
        min_gain.grid(row=6, column=2)
        # Asking the user for the maximum gain
        Label(Window, text="Max Gain", font=8).grid(row=6, column=3)
        max_gain = Entry(Window)
        max_gain.grid(row=6, column=4)
        # Label(Window, text="Max Gain", font=8).grid(row=6, column=3)
        # max_gain = Entry(Window)
        # max_gain.grid(row=6, column=4)
        # Asking the user for the gain factor vs Luminescence
        Label(Window, text="Gain factor vs Luminescence", font=8).grid(row=7, column=1)
        gain_factor = Entry(Window)
        gain_factor.grid(row=7, column=2)
        # Asking the user for the luminescence Exp/Gain threshold
        Label(Window, text="Luminescence Exp/Gain Threshold", font=8).grid(row=7, column=3)
        lum_threshold = Entry(Window)
        lum_threshold.grid(row=7, column=4)
        

        Button(Window, text = "Start", command = selections).grid(row=10, column=2)
        Button(Window, text = "Toggle Brightness Adjustment", command = toggle_exp).grid(row=10, column=3)


    def grab_thread(self, idx):
        global CHECKED_FRAME_0
        global CHECKED_FRAME_1

        runtime = sl.RuntimeParameters()
        frames_recorded = 0
        CHECKED_FRAME_0 = 10
        CHECKED_FRAME_1 = 15
        last_checked = [0]
        self.start = time.time()
        while self.exit_cam_app == False:
            start = time.time()
            if ((idx==1) and ((frames_recorded+1) % 2 == 0)):
                end = time.time()-start
                if (0.06 - end) > 0:
                    time.sleep(0.065 - end)
                frames_recorded += 1
                continue
            err = self.cams[idx].grab(runtime)
            if (err == sl.ERROR_CODE.SUCCESS):
                coincidence = False

                if random.random() < (1/15) or frames_recorded == 0:
                    coincidence = True
                    self.cams[idx].retrieve_image(self.cam_attributes[idx], sl.VIEW.LEFT)
                    img1 = cv2.resize(self.cam_attributes[idx].get_data(), (640, 360))
                    self.cams[idx].retrieve_image(self.cam_attributes[idx], sl.VIEW.RIGHT)
                    img2 = cv2.resize(self.cam_attributes[idx].get_data(), (640, 360))

                if self.toggle_auto_brightness == True:
                # Will need to be threaded as taking 0.15s per photo and also not every frame otherwise overlap

                    if idx == 0:
                        if (frames_recorded == CHECKED_FRAME_0) or (frames_recorded - last_checked[-1]>20):
                            if coincidence == False:
                                self.cams[idx].retrieve_image(self.cam_attributes[idx], sl.VIEW.LEFT)
                                img1 = cv2.resize(self.cam_attributes[idx].get_data(), (640, 360))
                            last_checked.append(frames_recorded)
                            threading.Thread(target=self.brightness_detector, args=(idx,)).start()
                    else:   
                        if (frames_recorded == CHECKED_FRAME_1) or (frames_recorded - last_checked[-1]>20):
                            if coincidence == False:
                                self.cams[idx].retrieve_image(self.cam_attributes[idx], sl.VIEW.LEFT)
                                img1 = cv2.resize(self.cam_attributes[idx].get_data(), (640, 360))
                            last_checked.append(frames_recorded)
                            threading.Thread(target=self.brightness_detector, args=(idx,)).start()

                    if (frames_recorded / self.fps == ((self.mins_p_chap/2)*60)) and (self.chapter_no[idx]==0) and (idx==1):
                        self.chapter(idx)
                        frames_recorded = 0
                        CHECKED_FRAME_1 = 15
                        last_checked = [0]
                    elif (frames_recorded / self.fps) == (self.mins_p_chap*60):
                        self.chapter(idx)
                        if idx == 0:
                            CHECKED_FRAME_0 = 10
                            frames_recorded = 0
                            last_checked = [0]
                        else:
                            CHECKED_FRAME_1 = 15
                            frames_recorded = 0
                            last_checked = [0]

                self.imgs_idvual[idx] = [img1, img2]
                frames_recorded += 1
            else:
                messagebox.showerror("error", "Error grabbing frame, camera may have crashed!")
            end = time.time()-start
            if (0.06 - end) > 0:
                time.sleep(0.06 - end)


    def set_recording_params(self, idx):
        print("enabling recording")
        folder = "recordings_test" # make this configurable??
        if not os.path.exists(folder+"/"):
            os.system("mkdir {}".format(folder))
        cam_num = {0:"front", 1:"rear"}
        val = self.cams[idx]
        # setting the recording parameters
        path_output = os.path.join(folder, "video_{}_cam_{}.svo".format(cam_num[idx], str(datetime.datetime.now().strftime(r"%d-%m_%H-%M-%S"))))
        recording_param = sl.RecordingParameters(path_output, sl.SVO_COMPRESSION_MODE.H265, self.fps)
        err = val.enable_recording(recording_param)
        if err != sl.ERROR_CODE.SUCCESS:
            print(err)
            messagebox.showerror("error", "Possible that output folder doesn't exist, look for 'recordings' folder in the same folder as this script")
            exit(1)


    def set_settings(self, idx, exp=False, gain=False):
        '''
        This method sets the settings of the camera
        '''
        val = self.cams[idx]
        cam = {0:"front", 1:"back"}

        if self.settings_change_front or self.settings_change_back:
            val.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, self.settings[idx]['Brightness'])
            val.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, self.settings[idx]['Contrast'])
            val.set_camera_settings(sl.VIDEO_SETTINGS.HUE, self.settings[idx]['Hue'])
            val.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, self.settings[idx]['Saturation'])
            val.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, self.settings[idx]['Sharpness'])
            val.set_camera_settings(sl.VIDEO_SETTINGS.GAMMA, self.settings[idx]['Gamma'])
            val.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO, 1)
            if self.min_gain < self.settings[idx]['Gain'] and self.max_gain > self.settings[idx]['Gain']:
                val.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, self.settings[idx]['Gain'])
            else:
                val.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, int((int(self.max_gain) + int(self.min_gain))/2))
            val.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, self.settings[idx]['Exposure'])
        else:
            if exp:
                if self.settings[idx]['Exposure'] == self.max_exp:
                    print("maximum exposure reached")
                    #messagebox.showinfo("Exposure Limit reached!", "Exposure has hit maximum limit in {} camera, adjust your maximum exposure value to fix luminence".format(cam[idx]))
                elif self.settings[idx]['Exposure'] == self.min_exp:
                    print("minimum exposure reached")
                    #messagebox.showinfo("Exposure Limit reached!", "Exposure has hit minimum limit in {} camera, adjust your minimum exposure value to fix luminence".format(cam[idx]))
                else:
                    print("changing {} exposure".format(cam[idx]))
                    if self.settings[idx]['Exposure'] > self.max_exp:
                        self.settings[idx]['Exposure'] = self.max_exp
                    elif self.settings[idx]['Exposure'] < self.min_exp:
                        self.settings[idx]['Exposure'] = self.min_exp
                    val.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, self.settings[idx]['Exposure'])
            if gain:
                if self.settings[idx]['Gain'] == self.max_gain:
                    print("maximum gain reached")
                    #messagebox.showinfo("Gain Limit reached!", "Gain has hit maximum limit in {} camera, adjust your maximum Gain value to fix luminence".format(cam[idx]))
                elif self.settings[idx]['Gain'] == self.min_gain:
                    print("minimum gain reached")
                    #messagebox.showinfo("Gain Limit reached!", "Gain has hit minimum limit in {} camera, adjust your minimum Gain value to fix luminence".format(cam[idx]))
                else:
                    print("changing {} gain".format(cam[idx]))
                    if self.settings[idx]['Gain'] > self.max_gain:
                        self.settings[idx]['Gain'] = self.max_gain
                    elif self.settings[idx]['Gain'] < self.min_gain:
                        self.settings[idx]['Gain'] = self.min_gain
                    val.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, self.settings[idx]['Gain'])

        if idx==0:
            self.settings_change_front = False
        elif idx==1:
            self.settings_change_back = False


    def brightness_detector(self, idx):
        img = self.imgs_idvual[idx][0]
        brightness = self.brightness(img, idx)
        dist_from_targ = abs(brightness - self.lum_targ)

        # another way around TESTING
        big_change = False
        small_change = False

        if (dist_from_targ > self.lum_threshold):
            big_change = True
        else:
            small_change = True

        if (brightness < self.lum_targ):
            if big_change:
                    # DARKER IMAGE
                if (self.settings[idx]['Exposure'] < self.max_exp):
                    self.settings[idx]['Exposure'] += 1
                    self.set_settings(idx, exp=True)
                elif (self.settings[idx]['Exposure'] == self.max_exp):
                    self.settings[idx]['Exposure'] += 0
                    self.set_settings(idx, exp=True)
                    if (self.settings[idx]['Gain'] < self.max_gain):
                        self.settings[idx]['Gain'] += int(dist_from_targ/self.gain_factor)
                        if (self.settings[idx]['Gain'] > self.max_gain):
                            self.settings[idx]['Gain'] = self.max_gain
                        self.set_settings(idx, gain=True)
                    elif (self.settings[idx]['Gain'] == self.max_gain):
                        self.settings[idx]['Gain'] += 0
                        self.set_settings(idx, gain=True)
            elif small_change:
                if (self.settings[idx]['Gain'] < self.max_gain):
                    self.settings[idx]['Gain'] += int(dist_from_targ/self.gain_factor)
                    if (self.settings[idx]['Gain'] > self.max_gain):
                        self.settings[idx]['Gain'] = self.max_gain
                    self.set_settings(idx, gain=True)
                elif (self.settings[idx]['Gain'] == self.max_gain):
                    self.settings[idx]['Gain'] += 0
                    self.set_settings(idx, gain=True)
        elif (brightness > self.lum_targ):
            if big_change:
                if (self.settings[idx]['Exposure'] > self.min_exp):
                    self.settings[idx]['Exposure'] -= 1
                    self.set_settings(idx, exp=True)
                elif (self.settings[idx]['Exposure'] == self.min_exp):
                    self.settings[idx]['Exposure'] -= 0
                    self.set_settings(idx, exp=True)
                    if (self.settings[idx]['Gain'] > self.min_gain):
                        self.settings[idx]['Gain'] -= int(dist_from_targ/self.gain_factor)
                        if (self.settings[idx]['Gain'] < self.min_gain):
                            self.settings[idx]['Gain'] = self.min_gain
                        self.set_settings(idx, gain=True)
                    elif (self.settings[idx]['Gain'] == self.min_gain):
                        self.settings[idx]['Gain'] -= 0
                        self.set_settings(idx, gain=True)
            elif small_change:
                if (self.settings[idx]['Gain'] > self.min_gain):
                    self.settings[idx]['Gain'] -= int(dist_from_targ/self.gain_factor)
                    if (self.settings[idx]['Gain'] < self.min_gain):
                        self.settings[idx]['Gain'] = self.min_gain
                    self.set_settings(idx, gain=True)
                elif (self.settings[idx]['Gain'] == self.min_gain):
                    self.settings[idx]['Gain'] -= 0
                    self.set_settings(idx, gain=True)
        
        else:
            self.settings[idx]['Exposure'] -= 0
            self.set_settings(idx, exp=False)
            self.settings[idx]['Gain'] -= 0
            self.set_settings(idx, gain=True)


        # self.csv_list[self.chapter_no[idx]][idx].append([time.time()-self.start, self.lum_targ, brightness, self.min_exp, self.max_exp, self.settings[idx]['Exposure'],
        #                                          self.min_gain, self.max_gain, self.settings[idx]['Gain'], self.settings[idx]['Brightness'], self.settings[idx]['Contrast'], self.settings[idx]['Hue'],
        #                                          self.settings[idx]['Saturation'], self.settings[idx]['Sharpness'], self.settings[idx]['Gamma']])


    def start_camera_thread(self):
        self.csv_list.append({0:[], 1:[]})
        threading.Thread(target=self.start_camera).start()


    def chapter(self, idx):
        print("chaptering!")
        self.chapter_no[idx] += 1

        val = self.cams[idx]
        val.disable_recording()

        # status = self.cams[idx].open(self.init)

        # if status != sl.ERROR_CODE.SUCCESS:
        #     print(repr(status))
        #     if str(repr(status)) == "LOW USB BANDWIDTH":
        #         messagebox.showerror("error", "'LOW USB BANDWIDTH', Try re-plugging in the cameras")
        #     self.root.destroy()
        #     sys.exit(0)

        self.set_recording_params(idx)

        if idx==1:
            self.csv_list.append({0:[], 1:[]})


    def start_camera(self):
        self.cams = [0,0]
        self.cam_attributes = [0,0]
        # defining the init parameters
        self.init = sl.InitParameters()
        self.init.camera_resolution = self.res
        self.init.depth_mode = sl.DEPTH_MODE.ULTRA # TODO: change to Neural
        self.init.camera_fps = self.fps
        self.init.coordinate_units = sl.UNIT.METER
        self.init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

        # Listing all the connected devices
        cameras = sl.Camera.get_device_list()

        if len(cameras) != 2:
            messagebox.showerror("error", "2 cameras could not be identified!")

        for idx, val in enumerate(cameras):
            if (val.serial_number == 37858896): #front cam
                index = 0
            elif (val.serial_number == 35054319): #back cam
                index = 1
            # set the input camera with specified serial number
            self.init.set_from_serial_number(val.serial_number)
            # create a camera object
            self.cams[index] = sl.Camera()
            self.cam_attributes[index] = sl.Mat()
            # apply init parameters to this camera
            status = self.cams[index].open(self.init)

            if status != sl.ERROR_CODE.SUCCESS:
                print(repr(status))
                if str(repr(status)) == "LOW USB BANDWIDTH":
                    messagebox.showerror("error", "'LOW USB BANDWIDTH', Try re-plugging in the cameras")
                self.root.destroy()
                sys.exit(0)

        # and we get the recording parameters called here - both cameras share the recording parameters
        for i in range(2):
            self.set_recording_params(i)

        try:
            self.settings
        except AttributeError:
            print("Attribute error while setting the config")
            self.settings = {0: {
                'Brightness': 4,
                'Contrast': 4,
                'Hue': 0,
                'Saturation': 4,
                'Sharpness': 8,
                'Gamma': 8,
                'Gain': 50,
                'Exposure': 5
            },
            1: {
                'Brightness': 4,
                'Contrast': 4,
                'Hue': 0,
                'Saturation': 4,
                'Sharpness': 8,
                'Gamma': 8,
                'Gain': 50,
                'Exposure': 5
            }
            }

        # set custom camera settings
        self.settings_change_front = True
        self.settings_change_back = True
        for i in range(2):
            self.set_settings(i)

        self.exit_cam_app = False
        self.imgs_idvual = {0:[], 1:[]}
        self.settings_change_front = False
        self.settings_change_back = False

        for idx, val in enumerate(self.cams):
            threading.Thread(target=self.grab_thread, args=(idx,)).start()
            time.sleep(0.1)

        time.sleep(1)

        while self.exit_cam_app == False:
            imgs = np.concatenate((np.concatenate(self.imgs_idvual[0], axis=1), np.concatenate(self.imgs_idvual[1], axis=1)), axis=0)
            cv2.imshow("ZED_Camera", imgs)
            # wait for 1 ms - seems needed to show image
            if self.settings_change_front:
                self.set_settings(0)
            elif self.settings_change_back:
                self.set_settings(1)

            cv2.waitKey(950)

        cv2.destroyAllWindows()
        for idx, val in enumerate(self.cams):
            val.disable_recording()
            val.close()


    def write_to_csv(self):
        # messagebox.showinfo("info", "Don't touch anything, writing to CSV file")
        # header = ['timestamp', 'lum_targ', 'lum', 'min_exp', 'max_exp', 'exp', 'min_gain', 'max_gain', 'gain', 'brightness', 'contrast', 'hue', 'saturation', 'sharpness', 'gamma']
        # for file in range(len(self.csv_list)):
        #     name = os.path.join('recordings', 'video_{}_details.csv'.format(time.time() - self.csv_list[file][0][-1][0]))
        #     with open(name, 'w', newline='') as f:
        #         writer = csv.writer(f)
        #         writer.writerow(header)
        #         writer.writerow(['front cam'])
        #         writer.writerows(self.csv_list[file][0])
        #         writer.writerow(['back cam'])
        #         writer.writerows(self.csv_list[file][1])
        # messagebox.showinfo("info", "CSV files created")
        self.root.destroy()


    def stop_camera(self):
        self.exit_cam_app = True


    def optimise_thread(self):
        messagebox.showinfo("info", "Only change settings if you have already started the camera")

        def change_settings_overcast():
            bright_val.set(6)
            contst_val.set(1)
            hue_val.set(0)
            sat_val.set(4)
            sharp_val.set(0)
            gamma_val.set(3)
            gain_val.set(35)
            exp_val.set(1)

            bright_val_back.set(6)
            contst_val_back.set(1)
            hue_val_back.set(0)
            sat_val_back.set(4)
            sharp_val_back.set(0)
            gamma_val_back.set(3)
            gain_val_back.set(35)
            exp_val_back.set(1)


        def change_settings_overcast_tree():
            bright_val.set(4)
            contst_val.set(2)
            hue_val.set(0)
            sat_val.set(4)
            sharp_val.set(5)
            gamma_val.set(8)
            gain_val.set(100)
            exp_val.set(1)

            bright_val_back.set(4)
            contst_val_back.set(2)
            hue_val_back.set(0)
            sat_val_back.set(4)
            sharp_val_back.set(5)
            gamma_val_back.set(8)
            gain_val_back.set(100)
            exp_val_back.set(1)


        def change_settings_sunny():
            bright_val.set(4)
            contst_val.set(2)
            hue_val.set(0)
            sat_val.set(4)
            sharp_val.set(5)
            gamma_val.set(8)
            gain_val.set(50)
            exp_val.set(3)

            bright_val_back.set(4)
            contst_val_back.set(2)
            hue_val_back.set(0)
            sat_val_back.set(4)
            sharp_val_back.set(5)
            gamma_val_back.set(8)
            gain_val_back.set(50)
            exp_val_back.set(3)


        def change_settings_sunny_tree():
            bright_val.set(4)
            contst_val.set(2)
            hue_val.set(0)
            sat_val.set(4)
            sharp_val.set(5)
            gamma_val.set(8)
            gain_val.set(50)
            exp_val.set(1)

            bright_val_back.set(4)
            contst_val_back.set(2)
            hue_val_back.set(0)
            sat_val_back.set(4)
            sharp_val_back.set(5)
            gamma_val_back.set(8)
            gain_val_back.set(50)
            exp_val_back.set(1)


        def change_settings_front_custom():
            # get all vals and save them to settings
            if exp_val.get() > self.max_exp or exp_val.get() < self.min_exp:
                messagebox.showerror("Error", "Exposure value out of range")
            else:
                self.settings[0] = {
                    "Brightness": bright_val.get(),
                    "Contrast": contst_val.get(),
                    "Hue": hue_val.get(),
                    "Saturation": sat_val.get(),
                    "Sharpness": sharp_val.get(),
                    "Gamma": gamma_val.get(),
                    "Gain": gain_val.get(),
                    "Exposure": exp_val.get(),
                }

                self.settings_change_front = True

        def change_settings_back_custom():
            if exp_val_back.get() > self.max_exp or exp_val_back.get() < self.min_exp:
                messagebox.showerror("Error", "Exposure value out of range")
            else:
                self.settings[1] = {
                    "Brightness": bright_val_back.get(),
                    "Contrast": contst_val_back.get(),
                    "Hue": hue_val_back.get(),
                    "Saturation": sat_val_back.get(),
                    "Sharpness": sharp_val_back.get(),
                    "Gamma": gamma_val_back.get(),
                    "Gain": gain_val_back.get(),
                    "Exposure": exp_val_back.get(),
                }

                self.settings_change_back = True

        nWindow = tk.Toplevel(self.root)
        nWindow.title("Setting Options")
        nWindow.geometry("900x700")
        tk.Label(nWindow, width=3).grid(column=0)
        tk.Label(nWindow, width=30).grid(row=0,column=2)

        Label(nWindow, text="Camera Settings Options", font=10).grid(row=1, column=3, columnspan=2)

        Label(nWindow, text="Front Cam", font=8).grid(row=2, column=2, columnspan=2)
        Label(nWindow, text="Brightness").grid(row=3, column=2)
        bright_val = tk.Scale(nWindow, from_=0, to=8, length=200,tickinterval=1, orient='horizontal')
        bright_val.grid(row=3, column=3)
        Label(nWindow, text="Contrast").grid(row=4, column=2)
        contst_val = tk.Scale(nWindow, from_=0, to=8, length=200,tickinterval=1, orient='horizontal')
        contst_val.grid(row=4, column=3)
        Label(nWindow, text="Hue").grid(row=5, column=2)
        hue_val = tk.Scale(nWindow, from_=0, to=11, length=200,tickinterval=2, orient='horizontal')
        hue_val.grid(row=5, column=3)
        Label(nWindow, text="Saturation").grid(row=6, column=2)
        sat_val = tk.Scale(nWindow, from_=0, to=8, length=200,tickinterval=1, orient='horizontal')
        sat_val.grid(row=6, column=3)
        Label(nWindow, text="Sharpness").grid(row=7, column=2)
        sharp_val = tk.Scale(nWindow, from_=0, to=8, length=200,tickinterval=1, orient='horizontal')
        sharp_val.grid(row=7, column=3)
        Label(nWindow, text="Gamma").grid(row=8, column=2)
        gamma_val = tk.Scale(nWindow, from_=1, to=9, length=200,tickinterval=1, orient='horizontal')
        gamma_val.grid(row=8, column=3)
        Label(nWindow, text="Gain").grid(row=9, column=2)
        gain_val = tk.Scale(nWindow, from_=0, to=100, length=200,tickinterval=25, orient='horizontal')
        gain_val.grid(row=9, column=3)
        Label(nWindow, text="Exposure").grid(row=10, column=2)
        exp_val = tk.Scale(nWindow, from_=0, to=100, length=200,tickinterval=25, orient='horizontal')
        exp_val.grid(row=10, column=3)

        Label(nWindow, text="Back Cam", font=8).grid(row=2, column=4, columnspan=2)
        Label(nWindow, text="Brightness").grid(row=3, column=4)
        bright_val_back = tk.Scale(nWindow, from_=0, to=8, length=200,tickinterval=1, orient='horizontal')
        bright_val_back.grid(row=3, column=5)
        Label(nWindow, text="Contrast").grid(row=4, column=4)
        contst_val_back = tk.Scale(nWindow, from_=0, to=8, length=200,tickinterval=1, orient='horizontal')
        contst_val_back.grid(row=4, column=5)
        Label(nWindow, text="Hue").grid(row=5, column=4)
        hue_val_back = tk.Scale(nWindow, from_=0, to=11, length=200,tickinterval=2, orient='horizontal')
        hue_val_back.grid(row=5, column=5)
        Label(nWindow, text="Saturation").grid(row=6, column=4)
        sat_val_back = tk.Scale(nWindow, from_=0, to=8, length=200,tickinterval=1, orient='horizontal')
        sat_val_back.grid(row=6, column=5)
        Label(nWindow, text="Sharpness").grid(row=7, column=4)
        sharp_val_back = tk.Scale(nWindow, from_=0, to=8, length=200,tickinterval=1, orient='horizontal')
        sharp_val_back.grid(row=7, column=5)
        Label(nWindow, text="Gamma").grid(row=8, column=4)
        gamma_val_back = tk.Scale(nWindow, from_=1, to=9, length=200,tickinterval=1, orient='horizontal')
        gamma_val_back.grid(row=8, column=5)
        Label(nWindow, text="Gain").grid(row=9, column=4)
        gain_val_back = tk.Scale(nWindow, from_=0, to=100, length=200,tickinterval=25, orient='horizontal')
        gain_val_back.grid(row=9, column=5)
        Label(nWindow, text="Exposure").grid(row=10, column=4)
        exp_val_back = tk.Scale(nWindow, from_=0, to=100, length=200,tickinterval=25, orient='horizontal')
        exp_val_back.grid(row=10, column=5)

        Button(nWindow, text="Confirm Front Selection", command=change_settings_front_custom, width=60).grid(row=11, column=2, columnspan=2)
        Button(nWindow, text="Confirm Back Selection", command=change_settings_back_custom, width=60).grid(row=11, column=4, columnspan=2)
        Label(nWindow, text="Presets", font=10).grid(row=12, column=3, columnspan=3)
        Button(nWindow, text="Sunny", command=change_settings_sunny, width=30).grid(row=13, column=3)
        Button(nWindow, text="Overcast", command=change_settings_overcast, width=30).grid(row=13, column=4)
        Button(nWindow, text="Overcast Tree Cover", command=change_settings_overcast_tree, width=30).grid(row=14, column=3)
        Button(nWindow, text="Sunny Tree Cover", command=change_settings_sunny_tree, width=30).grid(row=14, column=4)


    # potential checker to see if left and right cameras are producing the same quality image.
    def cam_check(self):
    #     # function to take care of changing atmosphere and even the settings.
    #     # score = brisque.score(image)
    #     # double check that score in both cameras is roughly the same
    #     self.double_cam_check = True
    #     self.take_picture()
        pass


    def brightness(self, im_file,camera_idx):

        global CHECKED_FRAME_0
        global CHECKED_FRAME_1

        start = time.time()  
        img = Image.fromarray(np.uint8(im_file))
        stat = ImageStat.Stat(img)
        r,g,b,null = stat.mean
        end = time.time()
        elapsed_time = end - start
        BRIGHTNESS_FREQUENCY = int((float(elapsed_time) * 10)*self.fps)
        if BRIGHTNESS_FREQUENCY < 10:
            BRIGHTNESS_FREQUENCY = 10
        
        if camera_idx == 0:
            CHECKED_FRAME_0 = int(CHECKED_FRAME_0 + BRIGHTNESS_FREQUENCY)
        else:
            CHECKED_FRAME_1 = int(CHECKED_FRAME_1 + BRIGHTNESS_FREQUENCY)
        

            
        return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))


if __name__ == '__main__':
    tkinterGUI = tkinterGUI()