import os
import shutil
from tqdm import tqdm

# defining functions
def reading_class(folder):
    """
    Reads all the txt files in the given folder and returns a dictionary with the file name as key and lines in the file as values
    """
    class_dict = dict() # Creating an empty dictionary
    for items in os.listdir(folder):
        if items.endswith("sses.txt"):
            count = 0
            with open(os.path.join(folder,items),'r') as f:
                for line in f:
                    class_dict[str(count)] = line.strip()
                    count = count + 1
    print(class_dict) # Prints the dictionary
    return(class_dict)


def get_key(val,dict_):
    for key, value in dict_.items():
         if val == value:
             return key
 
    return "key doesn't exist"

def creating_class_dict(classes_file):
    class_dict = dict()
    count = 0
    with open(classes_file,'r') as f:
        for line in f:
            # create a dictionary
            class_dict[str(count)] = line.strip()
            count = count + 1
    
    return(class_dict)

def replacing_class(label_txt, input_folder, out_folder, original_dict=None,target_dict=None):
    new_lines = []
    with open(os.path.join(input_folder,label_txt), 'r') as file_to_read:
        for line in file_to_read:
            class_name = original_dict[line.strip()[0]]
            target_number = get_key(val=class_name,dict_=target_dict)
            new_line = line.replace(line.strip()[0],str(target_number),1) 
            new_lines.append(new_line)
    #new_label_txt = label_txt[9:]
    with open(os.path.join(out_folder,label_txt), 'w') as filehandle:
        for listitem in new_lines:
            #filehandle.write('%s\n' % listitem)
            filehandle.write(listitem)
def replacing_class_disturbed(label_txt, input_folder, out_folder, original_dict=None,target_dict=None):
    new_lines = []
    with open(os.path.join(input_folder,label_txt), 'r') as file_to_read:
        for line in file_to_read:
            class_name = original_dict[line.strip()[0]]
            disturbed_id_ash = ["2","5","7"]
            if class_name in disturbed_id_ash:
                target_number = "2"
            elif class_name == "0":
                target_number = "1"
            else:
                target_number = get_key(val=class_name,dict_=target_dict)
            new_line = line.replace(line.strip()[0],str(target_number),1) 
            new_lines.append(new_line)
    #new_label_txt = label_txt[9:]
    with open(os.path.join(out_folder,label_txt), 'w') as filehandle:
        for listitem in new_lines:
            #filehandle.write('%s\n' % listitem)
            filehandle.write(listitem)

def removing_class(label_txt, input_folder, out_folder, target_classes=["0","2"]):
    new_lines = []
    with open(os.path.join(input_folder,label_txt), 'r') as file_to_read:
        for line in file_to_read:
            try:
                class_detection = line.strip()[0]
                if class_detection in target_classes:
                    new_lines.append(line)
            except:
                new_lines.append(line)
    #new_label_txt = label_txt[9:]
    with open(os.path.join(out_folder,label_txt), 'w') as filehandle:
        for listitem in new_lines:
            #filehandle.write('%s\n' % listitem)
            filehandle.write(listitem)

def main(folders,val_folder,test_folder,output_folder):
    for folder in folders:
        images_path = os.path.join(folder,"images")
        val_images = os.path.join(val_folder,"images")
        test_images = os.path.join(test_folder,"images")
        for item in os.listdir(images_path):
            unhashed_name = item[9:]
            val_ident = []
            for item2 in os.listdir(val_images):
                if unhashed_name == item2[9:]:
                    print("image in val set")
                    val_ident.append(item2)
                    break 

            if len(val_ident) == 0:
                test_ident = []
                for item3 in os.listdir(test_images):
                    if unhashed_name == item3[9:]:
                        print("image in val set")
                        test_ident.append(item3)
                        break
                if len(test_ident) == 0:
                    print("image not in val nor test set")
                    #we copy the image
                    shutil.copy(os.path.join(images_path,item),os.path.join(output_folder,"images",item))
                    # adapt the label
                    label_name = item[:-4]+".txt"
                    original_dict = reading_class(folder=folder)
                    target_dict = reading_class(folder=val_folder)
                    replacing_class(label_txt=label_name, input_folder=os.path.join(folder,"labels"), out_folder=os.path.join(output_folder,"labels"), original_dict=original_dict,target_dict=target_dict)

def main2(original_dict,target_dict,label_txt,out_folder):
    original_dict = creating_class_dict(original_dict)
    target_dict = creating_class_dict(target_dict)

    for item in os.listdir(label_txt):
        try:
            # replacing_class(label_txt=item, input_folder=label_txt, out_folder=out_folder, original_dict=original_dict,target_dict=target_dict)
            replacing_class_disturbed(label_txt=item, input_folder=label_txt, out_folder=out_folder, original_dict=original_dict,target_dict=target_dict)
        except:
            print("File not modified: "+item)
        
def copy_imgs_labels(lbl_folder, out_label,ori_img,out_img):

    for item in tqdm(os.listdir(lbl_folder)):

        if item.endswith(".txt"):
            #shutil.copyfile(os.path.join(lbl_folder,item),os.path.join(out_label,item))
            try:
                shutil.copy(os.path.join(ori_img,item[:-9]+".png"),os.path.join(out_img,item[:-9]+".png"))
            except:
                print("Not found or already exists: "+str(os.path.join(ori_img,item[:-4]+".png")))
                continue
        else:
            continue    

def keeping_only_classes(lbl_folder, out_folder):
    for item in tqdm(os.listdir(lbl_folder)):

        
        if item.endswith(".txt"):
            #shutil.copyfile(os.path.join(lbl_folder,item),os.path.join(out_label,item))
            removing_class(item, lbl_folder, out_folder, target_classes=["0","2"])
        else:
            continue    

def replace_numbers(label_txt,lbl_folder, out_folder):
    new_lines = []
    with open(os.path.join(lbl_folder,label_txt), 'r') as file_to_read:
        for line in file_to_read:
            try:
                if line.strip()[0] == "0":
                    new_lines.append(line)
                elif line.strip()[0] == "2":
                    new_line = line.replace(line.strip()[0],str(1),1) 
                    new_lines.append(new_line)
                else:
                    continue
            except:
                continue
    #new_label_txt = label_txt[9:]
    with open(os.path.join(out_folder,label_txt), 'w') as filehandle:
        for listitem in new_lines:
            #filehandle.write('%s\n' % listitem)
            filehandle.write(listitem)

def changing_classes(lbl_folder, out_folder):
    for item in tqdm(os.listdir(lbl_folder)):

        
        if item.endswith(".txt"):
            #shutil.copyfile(os.path.join(lbl_folder,item),os.path.join(out_label,item))
            replace_numbers(item, lbl_folder, out_folder)
        else:
            continue  

def copy_unmatched_imgs(input_folder,comparing_folder,out_folder):
    input_images = os.listdir(input_folder)
    comparing_images = os.listdir(comparing_folder)
    input_images_hash = False
    comparing_images_hash = True
    if input_images_hash: 
        input_images = [elem[9:] for elem in input_images]
    if comparing_images_hash: 
        comparing_images = [elem[9:] for elem in comparing_images]
    
    for item in input_images:
        if item not in comparing_images:
            shutil.copy(os.path.join(input_folder,item),os.path.join(out_folder,item))

def moving_images_no_label(input_img, input_lbl, out_img):

    all_labels = [elem[:-4] for elem in os.listdir(input_lbl)]

    for item in os.listdir(input_img):
        if item[:-4] not in all_labels:
            shutil.move(os.path.join(input_img,item),out_img)

def missing_images(input_img, input_label, ori_images,out_images):

    all_labels = [elem[:-4] for elem in os.listdir(input_label)]
    inp_images = [elem[:-4] for elem in os.listdir(input_img)]
    for item in all_labels:
        if item not in inp_images:
            if item.startswith("video"):
                shutil.copy(os.path.join(ori_images,item+".png"),os.path.join(out_images,item+".png"))
            else:    
                shutil.copy(os.path.join(ori_images,item[9:]+".png"),os.path.join(out_images,item+".png")) 

        
def remove_hash(input_img,input_lbl):
    img_to_rename = []
    lbl_to_rename = []
    end_hash = False
    for item in os.listdir(input_img):
        if not item.startswith("video"):
            if not item.startswith("Thum"):
                img_to_rename.append(item)
    if input_lbl != None:
        for item in os.listdir(input_lbl):
            if not item.startswith("video"):
                if not item.startswith("Thum"):
                    lbl_to_rename.append(item)
    for item in img_to_rename:
        try:
            os.rename(os.path.join(input_img,item),os.path.join(input_img,item[9:]))
        except FileExistsError:
            print("File already exists")
    if input_lbl != None:
        for item in lbl_to_rename:
            try:
                os.rename(os.path.join(input_lbl,item),os.path.join(input_lbl,item[9:]))
            except FileExistsError:
                print("File already exists")
    if end_hash:
        lbl_to_rename = []
        for item in os.listdir(input_lbl):
            if not item.startswith("Thum"):
                lbl_to_rename.append(item)
        for item in lbl_to_rename:
            try:
                os.rename(os.path.join(input_lbl,item),os.path.join(input_lbl,item[:-9]+".txt"))
            except FileExistsError:
                print("File already exists")




if __name__ == '__main__':
    lbl_folder = r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\Zed2i\training_sets\Glasgow\project-11-at-2022-11-11-12-12-1ccd6c70\labels"
    out_label = r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\Zed2i\training_sets\Glasgow\project-11-at-2022-11-11-12-12-1ccd6c70\new_labels"
    ori_img = r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\Zed2i\training_sets\Conwy\Conwy\ash_detection\slot_0\slot_0_images"
    out_img = r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\Zed2i\training_sets\Conwy\project-16778-at-2022-11-10-12-15-80536e40\images"

    # copy_imgs_labels(lbl_folder=lbl_folder, out_label=out_label,ori_img=ori_img,out_img=out_img)
    # keeping_only_classes(lbl_folder, out_label)
    # changing_classes(lbl_folder, out_label)
    # moving_images_no_label(input_img=r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\Labelled_training_sets\ash_detection\glasgow\still_images\images", input_lbl=r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\Labelled_training_sets\ash_detection\glasgow\still_images\labels", out_img=r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\Labelled_training_sets\ash_detection\glasgow\still_images\im_no_lbl")
    # missing_images(input_img=r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\Zed2i\training_sets\Glasgow\project-17246-at-2022-11-11-10-49-0d956bfd\images", input_label=r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\Zed2i\training_sets\Glasgow\project-17246-at-2022-11-11-10-49-0d956bfd\new_labels", ori_images=r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\Zed2i\training_sets\Glasgow\Glasgow\ash_detection\all_images",out_images=r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\Zed2i\training_sets\Glasgow\project-17246-at-2022-11-11-10-49-0d956bfd\im")
    remove_hash(input_img=r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\Labelled_training_sets\health_class\3",input_lbl=None)
    # main2(
    #     original_dict=r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\Zed2i\training_sets\Glasgow\project-11-at-2022-11-11-12-12-1ccd6c70\classes.txt",
    #     target_dict=r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\Labelled_training_sets\ash_detection\classes.txt",
    #     label_txt=lbl_folder,
    #     out_folder=out_label)
    # copy_unmatched_imgs(input_folder=r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\Zed2i\training_sets\Monmouth_2\ash_detection\slot_0\slot_0_images",comparing_folder=r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\Zed2i\training_sets\Monmouth_2\project-4-at-2022-10-18-12-43-c3a9d1bd\images",out_folder=r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\Zed2i\training_sets\Monmouth_2\no_ash")