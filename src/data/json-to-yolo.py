import os
import json
import sys
import shutil

target_dir = r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\GoPRO\health_classification\second_set_19_07"

dirs = []
top_dir = r"\\gb010587mm\IDA_datasets\Ash_Dieback\2021\GoPRO\health_classification"
sub_dirs = os.listdir(top_dir)
for i in sub_dirs:
    dirs.append(os.path.join(top_dir, i))

f = open(r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\project-8-at-2022-07-19-09-29-87050255.json")
data = json.load(f)

dieback_lvl = {"No dieback":"100-76", r"100% - 75% live crown":"100-76", r"75% - 50% live crown":"75-51", r"50% - 25% live crown":"50-26", r"25% - 0% live crown":"25-0", "Dead":"25-0"}
os.system("mkdir undefined")

for dir in dirs:
    files_for_data = [i.replace("(", "").replace(")", "").replace(",", "") for i in os.listdir(dir)]
    images = os.listdir(dir)
    for element in data:
        if element['data']['image'].split("/")[0] == 'azure-blob:':
            if element['data']['image'].split("/")[-1] in images:
                try:
                    shutil.copyfile(os.path.join(dir, element['data']['image'].split("/")[-1]), os.path.join(target_dir, dieback_lvl[element['annotations'][0]['result'][0]['value']['choices'][0]], element['data']['image'].split("/")[-1]))
                    #os.system("cp {} {}".format(os.path.join(dir, element['data']['image'].split("/")[-1]), os.path.join(target_dir, dieback_lvl[element['annotations'][0]['result'][0]['value']['choices'][0]])))
                except IndexError:
                    shutil.copyfile(os.path.join(dir, element['data']['image'].split("/")[-1]), os.path.join(target_dir, "undefined", element['data']['image'].split("/")[-1]))
                    #os.system("cp {} {}".format(os.path.join(dir, element['data']['image'].split("/")[-1]), os.path.join(target_dir, "undefined")))
        # if element['data']['image'].split("/")[1] == 'data':
        #     file = "-".join(element['data']['image'].split("/")[-1].split("-")[1:])
        #     if file in files_for_data:
        #         idx = files_for_data.index(file)
        #         try:
        #             os.system("cp {} {}".format(os.path.join(dir, images[idx]), os.path.join(target_dir, dieback_lvl[element['annotations'][0]['result'][0]['value']['choices'][0]])))
        #         except IndexError:
        #             os.system("cp {} {}".format(os.path.join(dir, images[idx]), os.path.join(target_dir, "undefined")))