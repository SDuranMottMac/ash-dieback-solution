# importing libraries
import json
import os
import cv2

class video2labels():
    """
    This class generates an object that represents the training data from the video labelled
    """
    def __init__(self,video_folder,json_labels,out_directory):
        self.video_folder = video_folder
        self.json_labels = json_labels
        self.out_dir = out_directory
        self.class_dict = {"Other Dead Trees":"0","Distant Ash Tree":"1","Ash tree":"2","Immature Ash Tree":"3"}
    
    def setting_outdir(self):
        # creating folders in the output directory
        os.makedirs(os.path.join(self.out_dir,"images"),exist_ok=True)
        os.makedirs(os.path.join(self.out_dir,"labels"),exist_ok=True)
    
    def reading_json(self):
        with open(self.json_labels, 'r') as f:
            video_labels = json.load(f)
        return(video_labels)
    
    def convert_from_ls(self,x,y,w,h,img_width,img_height):
        x_f = img_width * x / 100.0
        y_f = img_height * y / 100.0
        w_f = img_width * w / 100.0
        h_f = img_height * h / 100.0

        return(x_f, y_f,w_f,h_f)
    
    def eliminate_duplicate_labels(self):
        files_to_remove = []
        for item in os.listdir(os.path.join(self.out_dir,"labels")):
            # appending the file to the list
            files_to_remove.append(os.path.join(self.out_dir,"labels",item))
            # reading label file
            current_label = open(os.path.join(self.out_dir,"labels",item))
            data_label = current_label.read()
            # replacing end splitting the text when new line is seen
            data_label_list = data_label.split("\n")
            # closing the file
            current_label.close()
            # removing duplicates from list
            final_list = [*set(data_label_list)]
            # saving the new file
            new_filename = item[:-4]+"_new.txt"
            # open file in write mode
            with open(os.path.join(os.path.join(self.out_dir,"labels",new_filename)), 'w') as fp:
                for line in final_list:
                    if len(line) < 2:
                        continue
                    # write each item on a new line
                    fp.write("%s\n" % line)
                print('Done')
        
        # removing old files
        for item in files_to_remove:
            os.remove(item)
        
        # renaming new items
        for item in os.listdir(os.path.join(self.out_dir,"labels")):
            os.rename(os.path.join(self.out_dir,"labels",item),os.path.join(self.out_dir,"labels",item[:-8]+".txt"))
    
    def obtaining_sequence_data(self, sequence_dict):
        # obtaining video name
        video_name = os.path.basename(sequence_dict["data"]["video"].split("//")[-1][:-4])
        print(video_name)
        print(os.path.isfile(os.path.join(self.video_folder,video_name+".mp4")))
        # obtaining sequence list
        if len(sequence_dict["annotations"]) > 0:
            result_list = sequence_dict["annotations"][0]["result"]
            # looping through result items
            for i in range(len(result_list)):
                print("Starting result block "+str(i))
                for j in range(len(result_list[i]["value"]["sequence"])):
                    print("Going for sequence "+str(j))
                    if j == 0:
                        # saving corresponding label
                        # =========================
                        # obtaining frames
                        frame_number = int(result_list[i]["value"]["sequence"][j]["time"]*15)
                        print(frame_number)
                        # reading video
                        video = cv2.VideoCapture(os.path.join(self.video_folder,video_name+".mp4"))
                        # extracting frame
                        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                        ret, frame = video.read()
                        if ret != True:
                            continue
                        # obtaining class
                        try:
                            class_detected = str(result_list[i]["value"]["labels"][0])
                            class_number = self.class_dict[class_detected]
                        except:
                            class_number = "1"
                        # obtaining shape of image
                        img_width = frame.shape[1]
                        img_height = frame.shape[0]
                        # obtaining bbox
                        x = result_list[i]["value"]["sequence"][j]["x"]
                        y = result_list[i]["value"]["sequence"][j]["y"]
                        width = result_list[i]["value"]["sequence"][j]["width"]
                        height = result_list[i]["value"]["sequence"][j]["height"]
                        # checking whether there are negative labels
                        if (x < 0) or (y < 0):
                            continue
                        #transforming from ls format to pixels
                        x,y,width,height = self.convert_from_ls(x,y,width,height,img_width,img_height)
                        #calculating centre x y
                        x = x + width/2
                        y = y + height/2
                        # normalising data
                        x = abs(x/img_width)
                        y = abs(y/img_height)
                        width = abs(width/img_width)
                        height = abs(height/img_height)
                        # saving it 
                        lbl_name = video_name+"_"+str(frame_number)+".txt"
                        if os.path.isfile(os.path.join(self.out_dir,"labels",lbl_name)):
                            # Open a file with access mode 'a'
                            file_object = open(os.path.join(self.out_dir,"labels",lbl_name), 'a')
                            # Append 'hello' at the end of file
                            line_to_append = class_number+" "+str(x)+" "+str(y)+" "+str(width)+" "+str(height)+"\n"
                            file_object.write(line_to_append)
                            # Close the file
                            file_object.close()
                        else:
                            # Open a file with access mode 'a'
                            file_object = open(os.path.join(self.out_dir,"labels",lbl_name), 'w')
                            # Append 'hello' at the end of file
                            line_to_append = class_number+" "+str(x)+" "+str(y)+" "+str(width)+" "+str(height)+"\n"
                            file_object.write(line_to_append)
                            # Close the file
                            file_object.close()
                        
                        # saving first image
                        # ==================
                        # saving image
                        img_name = video_name+"_"+str(frame_number)+".png"
                        if os.path.isfile(os.path.join(self.out_dir,"images",img_name)):
                            print("Image already saved")
                        else:
                            cv2.imwrite(os.path.join(self.out_dir,"images",img_name), frame)
                        
                    else:
                        # saving corresponding label
                        # =========================
                        # frame number
                        frame_number = int(result_list[i]["value"]["sequence"][j]["time"]*15)
                        previous_frame_number = int(result_list[i]["value"]["sequence"][j-1]["time"]*15)
                        print(frame_number)
                        # reading video
                        video = cv2.VideoCapture(os.path.join(self.video_folder,video_name+".mp4"))
                        # extracting frame
                        print("extracting frame")
                        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                        print("reading video")
                        ret, frame = video.read()
                        if ret != True:
                            continue
                        # obtaining class
                        try:
                            class_detected = str(result_list[i]["value"]["labels"][0])
                            class_number = self.class_dict[class_detected]
                        except:
                            class_number = "1"
                        # obtaining shape of image
                        img_width = frame.shape[1]
                        img_height = frame.shape[0]
                        # obtaining bbox
                        x = result_list[i]["value"]["sequence"][j]["x"]
                        y = result_list[i]["value"]["sequence"][j]["y"]
                        width = result_list[i]["value"]["sequence"][j]["width"]
                        height = result_list[i]["value"]["sequence"][j]["height"]
                        # checking that there are no negative labels
                        if (x < 0) or (y < 0):
                            continue
                        # transforming from ls format to pixels
                        x,y,width,height = self.convert_from_ls(x,y,width,height,img_width,img_height)
                        #calculating centre x y
                        x = x + width/2
                        y = y + height/2
                        # normalising data
                        x = abs(x/img_width)
                        y = abs(y/img_height)
                        width = abs(width/img_width)
                        height = abs(height/img_height)
                        # saving it 
                        lbl_name = video_name+"_"+str(frame_number)+".txt"
                        if os.path.isfile(os.path.join(self.out_dir,"labels",lbl_name)):
                            # Open a file with access mode 'a'
                            file_object = open(os.path.join(self.out_dir,"labels",lbl_name), 'a')
                            # Append 'hello' at the end of file
                            line_to_append = class_number+" "+str(x)+" "+str(y)+" "+str(width)+" "+str(height)+"\n"
                            file_object.write(line_to_append)
                            # Close the file
                            file_object.close()
                        else:
                            # Open a file with access mode 'a'
                            file_object = open(os.path.join(self.out_dir,"labels",lbl_name), 'w')
                            # Append 'hello' at the end of file
                            line_to_append = class_number+" "+str(x)+" "+str(y)+" "+str(width)+" "+str(height)+"\n"
                            file_object.write(line_to_append)
                            # Close the file
                            file_object.close()
                        
                        # saving first image
                        # ==================
                        # saving image
                        img_name = video_name+"_"+str(frame_number)+".png"
                        if os.path.isfile(os.path.join(self.out_dir,"images",img_name)):
                            print("Image already saved")
                        else:
                            cv2.imwrite(os.path.join(self.out_dir,"images",img_name), frame)
                        
                        # saving all intermediate frames and detections
                        # =============================================
                        intermediate_steps = frame_number - previous_frame_number
                        # coordinates
                        x2 = x
                        y2 = y
                        w2 = width
                        h2 = height
                        x1 = result_list[i]["value"]["sequence"][j-1]["x"]
                        y1 = result_list[i]["value"]["sequence"][j-1]["y"]
                        w1 = result_list[i]["value"]["sequence"][j-1]["width"]
                        h1 = result_list[i]["value"]["sequence"][j-1]["height"]
                        # checking that there are no negative labels
                        if (x1 < 0) or (y1 < 0):
                            continue
                        # transforming previous data to pixels
                        x1,y1,w1,h1 = self.convert_from_ls(x1,y1,w1,h1,img_width,img_height)
                        #calculating centre x y
                        x1 = x1 + w1/2
                        y1 = y1 + h1/2
                        # normalising data
                        x1 = abs(x1/img_width)
                        y1 = abs(y1/img_height)
                        w1 = abs(w1/img_width)
                        h1 = abs(h1/img_height)
                        # proportionality
                        diffX = x2 - x1
                        diffY = y2 - y1
                        diffW = w2 - w1
                        diffH = h2 - h1
                        for m in range(intermediate_steps):
                            if m == 0:
                                continue
                            # intermediate coordinates
                            x_int = x1 + (m*diffX/intermediate_steps)
                            y_int = y1 + (m*diffY/intermediate_steps)
                            # intermediate width/height
                            w_int = w1 + (m*diffW/intermediate_steps)
                            h_int = h1 + (m*diffH/intermediate_steps) 
                            
                            # saving corresponding label
                            # =========================
                            frame_number = int(result_list[i]["value"]["sequence"][j-1]["time"]*15)+m
                            print(frame_number)
                            # reading video
                            video = cv2.VideoCapture(os.path.join(self.video_folder,video_name+".mp4"))
                            # extracting frame
                            video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                            ret, frame = video.read()
                            if ret != True:
                                continue
                            # obtaining class
                            try:
                                class_detected = str(result_list[i]["value"]["labels"][0])
                                class_number = self.class_dict[class_detected]
                            except:
                                class_number = "1"
                            # saving it 
                            lbl_name = video_name+"_"+str(frame_number)+".txt"
                            if os.path.isfile(os.path.join(self.out_dir,"labels",lbl_name)):
                                # Open a file with access mode 'a'
                                file_object = open(os.path.join(self.out_dir,"labels",lbl_name), 'a')
                                # Append 'hello' at the end of file
                                line_to_append = class_number+" "+str(x_int)+" "+str(y_int)+" "+str(w_int)+" "+str(h_int)+"\n"
                                file_object.write(line_to_append)
                                # Close the file
                                file_object.close()
                            else:
                                # Open a file with access mode 'a'
                                file_object = open(os.path.join(self.out_dir,"labels",lbl_name), 'w')
                                # Append 'hello' at the end of file
                                line_to_append = class_number+" "+str(x_int)+" "+str(y_int)+" "+str(w_int)+" "+str(h_int)+"\n"
                                file_object.write(line_to_append)
                                # Close the file
                                file_object.close()
                            # saving  image
                            # ==================
                            # saving image
                            img_name = video_name+"_"+str(frame_number)+".png"
                            if os.path.isfile(os.path.join(self.out_dir,"images",img_name)):
                                print("Image already saved")
                            else:
                                cv2.imwrite(os.path.join(self.out_dir,"images",img_name), frame)
    
    def generating_labels(self,video_labels):
        for ix,item in enumerate(video_labels):
            print("Dealing with item: "+str(ix)+" out of "+str(len(video_labels))+" items")
            self.obtaining_sequence_data(item)

if __name__ == "__main__":

    # input data
    video_folder = r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\Zed2i\training_sets\N_Yorkshire\all_vids" # folder containing all mp4 videos
    json_labels = r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\Zed2i\training_sets\temp\export_14135_project-14135-at-2022-11-04-08-33-f5537aa7.json" # json file containing all the labels downloaded from label studio
    out_directory = r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\Zed2i\training_sets\temp\videos_nyorks_final" # directory to save images and labels
    # processing data
    # instantiating the project
    Conwy = video2labels(video_folder,json_labels,out_directory)
    # setting out directory
    Conwy.setting_outdir()
    # json video labels
    json_labels = Conwy.reading_json()
    # generating the labels
    Conwy.generating_labels(json_labels)
    # eliminating duplicates detections in labels
    Conwy.eliminate_duplicate_labels()




                        
                        



