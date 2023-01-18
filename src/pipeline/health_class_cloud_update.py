import os
import shutil

folder1 = r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\GoPRO\health_classification\25-0"
folder2 = r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\GoPRO\health_classification\50-26"
folder3 = r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\GoPRO\health_classification\75-51"
folder4 = r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\GoPRO\health_classification\100-76"

label_stud_list = os.listdir(folder1) + os.listdir(folder2) + os.listdir(folder3) + os.listdir(folder4)

folder = r"\\gb010587mm\IDA_datasets\Ash_Dieback\2021\GoPRO\health_classification"

for fol in os.listdir(folder):
    files = os.listdir(os.path.join(folder, fol))
    for file in files:
        if file.endswith(".jpg"):
            if file in label_stud_list:
                pass
            else:
                shutil.copy(os.path.join(folder, fol, file), os.path.join(folder, "upload", file))