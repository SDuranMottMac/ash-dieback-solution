# INFO ON PIPELINE
auth = 'Token 4a2c9364707d268eeafd45db8e0fe0d4a4ef9b43'
# BUFFER

import os
import json
import requests
import argparse
import shutil

def remove_dups(project_no):
    dups_to_remove = {}
    no_of_tasks = requests.get("http://localhost:8080/api/projects/{}".format(project_no), headers={'Authorization':auth}).json()['task_number']
    r = requests.get("http://localhost:8080/api/projects/{}/tasks/".format(project_no), headers={'Authorization':auth}, params={'page_size':no_of_tasks})

    for img in r.json():
        dups_to_remove[img['id']] = img['data']['image']

    to_remove = []
    vals = list(dups_to_remove.values())
    for key, val in sorted(dups_to_remove.items()):
        vals.remove(val)
        if val in vals:
            to_remove.append(key)

    for task in to_remove:
        r = requests.delete("http://localhost:8080/api/tasks/{}".format(task), headers={'Authorization':auth})


def create_upload_json(classification, detection, lbl_path, project_no):
    if detection:
        if project_no == 7:
            container_name = "ashtreeblob"
        elif project_no == 5:
            container_name = "ashtreeblob-dave"
    elif classification:
        container_name = "ashtreeblob-health-587mm"

    azure_stores = requests.get("http://localhost:8080/api/storages/azure", headers={'Authorization':auth}, params={'project':project_no}).json()
    for id, val in enumerate(azure_stores):
        if val['container'] == container_name:
            azure_storage_no = azure_stores[id]['id'] # reference to the container holding the images for this label studio bucket, id retrieved from label studio side.
    r = requests.post('http://localhost:8080/api/storages/azure/{}/sync'.format(azure_storage_no), headers={'Authorization':auth})

    if r.status_code != 200:
        raise ValueError('Azure container did not sync with the label studio bucket.')

    lbl_ROOT = lbl_path # r"\\localhost\IDA_datasets\Ash_Dieback\2022\GoPRO\Raw_split\Maximising_postcodes\dataset2\labels"
    no_of_tasks = requests.get("http://localhost:8080/api/projects/{}".format(project_no), headers={'Authorization':auth}).json()['task_number']
    r = requests.get("http://localhost:8080/api/projects/{}/tasks/".format(project_no), headers={'Authorization':auth}, params={'page_size':no_of_tasks})

    img_width = 4000
    img_height = 3000

    azure_img_container = r.json()
    lbl_dir = os.listdir(lbl_ROOT)
    data = []

    ### ADD FUNCTIONALITY TO CONSIDER ADDING MORE LABELS TO EXISTING DATASET WITHOUT DUPLICATING
    ### NEED A FLAG FOR WHETHER A LABEL IS A HEALTH CLASS OR BOUNDING BOX
    count = 0
    for img in azure_img_container:

            data_tmp = {
            "data": {
                "image": "{}".format(img['data']['image'])
                },
            "predictions": [{
                "result": [
                    # need to iterate over result for multiple detections in one photo
                ]
                }],
                "score": 0
            }

            if detection:
                lbl_file = img['data']['image'].split("/")[-1][:-3]+"txt"

                if lbl_file in lbl_dir:
                    tmp_file = open(os.path.join(lbl_ROOT, lbl_file), 'r')

                    for line in tmp_file.readlines():
                        count = 0
                        obj_class, obj_x, obj_y, obj_w, obj_h = line[:-1].split(" ")
                        obj_x, obj_w, obj_y, obj_h = float(obj_x)*100, float(obj_w)*100, float(obj_y)*100, float(obj_h)*100
                        obj_x = obj_x - 0.5*obj_w
                        obj_y = obj_y - 0.5*obj_h

                        data_tmp["predictions"][0]["result"].append({
                            "id": "{}".format(count),
                            "type": "rectanglelabels",
                            "from_name": "label", "to_name": "image",
                            "original_width": img_width, "original_height": img_height,
                            "image_rotation": 0,
                            "value": {
                            "rotation": 0,
                            "x": obj_x, "y": obj_y,
                            "width": obj_w, "height": obj_h,
                            "rectanglelabels": ["Ash tree"]
                            }
                        })
                        count+=1

            elif classification:
                dieback_lvl = {0: "No dieback", 1:r"100% - 75% live crown", 2:r"75% - 50% live crown", 3:r"50% - 25% live crown", 4:r"25% - 0% live crown", 5:"Dead"}
                for folder in lbl_dir:
                    jpgs = os.listdir(os.path.join(lbl_ROOT, folder))
                    if img['data']['image'].split("/")[-1] in jpgs:
                        count = 0
                        data_tmp["predictions"][0]["result"].append({
                            "id": "{}".format(count),
                            "type": "choices",
                            "from_name": "choice", "to_name": "image",
                            "value": {
                                "choices": ["{}".format(dieback_lvl[int(folder)])]
                            }
                        })
                        count+=1

            else:
                raise ValueError('No detection or classification flag set.')

            data.append(data_tmp)

    # finally need to write data out to a json file
    json_file = json.dumps(data)

    # TODO: FINAL STEP WOULD BE TO AUTOMATE UPLOAD BUT RAN INTO SOME ISSUES
    #
    # r = requests.post('http://gb010587aa:8080/api/projects/{}/import'.format(project_no), headers={'Authorization':auth}, params={'data':data[0]})
    # if r.status_code == 201:
    #     print("success!")
    # elif r.status_code != 201:
    #     print("tasks not uploaded")

    with open(r"\\localhost\Software_dev\Ash_Dieback_Solution\ash_dieback_solution\src\data\label_studio.json", 'w') as f:
        f.write(json_file)

def create_dirtree_without_files(src, dst):
   
      # getting the absolute path of the source
    # directory
    src = os.path.abspath(src)
     
    # making a variable having the index till which
    # src string has directory and a path separator
    src_prefix = len(src) + len(os.path.sep)
     
    # making the destination directory
    os.makedirs(dst,exist_ok=True)
     
    # doing os walk in source directory
    for root, dirs, files in os.walk(src):
        for dirname in dirs:
           
            # here dst has destination directory,
            # root[src_prefix:] gives us relative
            # path from source directory
            # and dirname has folder names
            dirpath = os.path.join(dst, root[src_prefix:], dirname)
             
            # making the path which we made by
            # joining all of the above three
            os.makedirs(dirpath,exist_ok=True)

def mimic_folder_structure(input_fold, output_dir,file_ext =".txt"):
    """
    This function mimics the structure of the input directory and copy only the files of interest
    """
    # first, mimicking the structure of the input directory
    create_dirtree_without_files(input_fold,output_dir)
    # copying only the txt files
    for root, dirs, files in os.walk(input_fold):
        for item in files:
            if item.endswith(file_ext):
                rel_path = os.path.relpath(root,input_fold)
                shutil.copy(os.path.join(root,item),os.path.join(output_dir,rel_path,item))






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--upload', help='this will run the create_upload_json function', action='store_true')
    parser.add_argument('--lbl_path', type=str, help='paths to the labels you want to upload')
    parser.add_argument('--project_no', type=int, required=False, help='project number related to the block of images being labelled, this can be found in the URL of the label studio project')
    parser.add_argument('--drop_dups', help='this runs the duplicate removal function', action='store_true')
    parser.add_argument('--classification', help='this will run the create_upload_json function', action='store_true')
    parser.add_argument('--detection', help='this will run the create_upload_json function', action='store_true')
    parser.add_argument('--azure_txt', help='this will prepare directory to be uploaded to azure', action='store_true')
    args = parser.parse_args()

    if args.upload:
        create_upload_json()
    if args.drop_dups:
        remove_dups()

    if args.azure_txt:
        input_dir =r"\\gb002339ab\One_Touch\GB010587MM\Conwy 22"
        out_dir=r"\\gb010587mm\Seagate_one_touch\GB002339AB\conwyfiles"
        # mimicking folder structure
        mimic_folder_structure(input_dir, out_dir,file_ext =".txt")