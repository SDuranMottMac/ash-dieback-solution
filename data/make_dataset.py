# -*- coding: utf-8 -*-
import os 
import pandas as pd
import shutil
from sklearn.utils import shuffle


def importing_data(input_datainfo, input_county):
    ''' importing metadata about the datasets'''
    df_info = pd.read_csv(input_datainfo,sep=";",encoding="utf8",encoding_errors='ignore',header=None)
    df_county = pd.read_csv(input_county,sep=";",encoding="utf8",encoding_errors='ignore',header=None)

    return(df_info,df_county)

def reading_images(all_images,unlabelled):
    all_data = dict()
    all_images = os.listdir(all_images)
    unlabelled = os.listdir(unlabelled)

    all_data["all_images"] = all_images
    all_data["unlabelled"] = unlabelled

    return(all_data)

def underepresented_postcodes(df_county,n_under = 38):
    under_represented = []

    for index, row in df_county.iterrows():
        if (row[1] < n_under):
            under_represented.append(row[0])
    
    return(under_represented)

def make_dataset_(all_data,df_county,df_info,under_represented,outfolder,max_postcodes = 1000,dataset_size = 5000):
    # we'll create a dictionary to keep track of the movement
    tracking = dict()
    for index, row in df_county.iterrows():
        tracking[row[0]] = list()
    print(tracking)
    # first, we need to move the under_represented data
    for postcode in under_represented:
        for index, row in df_info.iterrows():

            if (row[4] == postcode):
                if row[1] not in tracking[postcode]:
                    tracking[postcode].append(row[1])

                    if row[1] in all_data["all_images"]:
                        shutil.copy(os.path.join(all_images_jpg,row[1]),os.path.join(outfolder,"images",row[1]))
                        label_name = row[1][:-4]+".txt"
                        try:
                            shutil.copy(os.path.join(all_images_label,label_name),os.path.join(outfolder,"labels",label_name))
                        except:
                            pass
                    
                    elif row[1] in all_data["unlabelled"]:
                        shutil.copy(os.path.join(unlabelled_jpg,row[1]),os.path.join(outfolder,"images",row[1]))
    

    # we copy all of the rest
    df_shuffled = shuffle(df_info)
    #print(df_shuffled.head(10))
    for index, row in df_shuffled.iterrows():
        
        # checking whether we need more data from this postcode
        if row[4] in under_represented:
            continue
        elif (len(tracking[row[4]])>= max_postcodes):
            continue
        else:
            if row[1] in tracking[row[4]]:
                continue
            else:
                tracking[row[4]].append(row[1])
                if row[1] in all_data["all_images"]:
                    shutil.copy(os.path.join(all_images_jpg,row[1]),os.path.join(outfolder,"images",row[1]))
                    label_name = row[1][:-4]+".txt"
                    try:
                        shutil.copy(os.path.join(all_images_label,label_name),os.path.join(outfolder,"labels",label_name))
                    except:
                        pass
                
                elif row[1] in all_data["unlabelled"]:
                    shutil.copy(os.path.join(unlabelled_jpg,row[1]),os.path.join(outfolder,"images",row[1]))
        if len(os.listdir(os.path.join(outfolder,"images"))) == dataset_size:
            break

def make_dataset_final(all_data,df_county,df_info,outfolder,max_postcodes = 1000,dataset_size = 5000):
    # we'll create a dictionary to keep track of the movement
    tracking = dict()
    for index, row in df_county.iterrows():
        tracking[row[0]] = list()
    print(tracking)
  
    # we copy all of the rest
    df_shuffled = shuffle(df_info)
    #print(df_shuffled.head(10))
    for index, row in df_shuffled.iterrows():
        
        # checking whether we need more data from this postcode
        
        if (len(tracking[row[4]])>= max_postcodes):
            continue
        else:
            if row[1] in tracking[row[4]]:
                continue
            else:
                tracking[row[4]].append(row[1])
                if row[1] in all_data["all_images"]:
                    shutil.copy(os.path.join(all_images_jpg,row[1]),os.path.join(outfolder,"images",row[1]))
                    label_name = row[1][:-4]+".txt"
                    try:
                        shutil.copy(os.path.join(all_images_label,label_name),os.path.join(outfolder,"labels",label_name))
                    except:
                        pass
                
                elif row[1] in all_data["unlabelled"]:
                    shutil.copy(os.path.join(unlabelled_jpg,row[1]),os.path.join(outfolder,"images",row[1]))
        if len(os.listdir(os.path.join(outfolder,"images"))) > dataset_size:
            break

def new_txtfile(older_dfinfo,output_folder):
    dataset1 = os.listdir(os.path.join(output_folder,"images"))
    drop_index = []
    for index, row in older_dfinfo.iterrows():
        if row[1] in dataset1:
            drop_index.append(index)
    
    new_df = older_dfinfo.drop(drop_index)
    name = "dataset_without.csv"
    new_df.to_csv(os.path.join(output_folder,name), sep=";",header=None,index=False)

    return(new_df)

def new_txtfile2(older_dfinfo,output_folder):
    dataset1 = os.listdir(os.path.join(output_folder,"images"))
    drop_index = []
    for index, row in older_dfinfo.iterrows():
        if row[1] not in dataset1:
            drop_index.append(index)
    
    new_df = older_dfinfo.drop(drop_index)
    name = "dataset.csv"
    new_df.to_csv(os.path.join(output_folder,name),sep=";",header=None,index=False)

def count_df_new(new_df,output_folder):

    postcodes = new_df[4].unique().tolist()
    counts = []
    for post in postcodes:
        counts.append(len(new_df[new_df[4] == post]))
    
    new_counts = list(zip(postcodes,counts))
    new_counts_df = pd.DataFrame(new_counts, columns=['Month','Day'])
    name = "df_counts.txt"
    new_counts_df.to_csv(os.path.join(output_folder,name),sep=";",header=None,index=False)


if __name__ == '__main__':

    all_images_jpg = r'\\gb010587mm\IDA_datasets\Ash_Dieback\2021\GoPRO\ash_detection\all_images'
    all_images_label = r'\\gb010587mm\IDA_datasets\Ash_Dieback\2021\GoPRO\ash_detection\all_labels'
    unlabelled_jpg = r'\\gb010587mm\IDA_datasets\Ash_Dieback\2022\GoPRO\Unlabelled'
    
    # we need to loop over them all

    folder_previous = [r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\GoPRO\Raw\dataset7",r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\GoPRO\Raw\dataset8",r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\GoPRO\Raw\dataset9",r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\GoPRO\Raw\dataset10"]

    for i in range(len(folder_previous)-1):


        df_info,df_county = importing_data(input_datainfo=os.path.join(folder_previous[i],"dataset_without.csv"), input_county=os.path.join(folder_previous[i],"df_counts.txt"))
        print(df_info.head(10))
        under_represented = underepresented_postcodes(df_county)
        all_data=reading_images(all_images=r"\\gb010587mm\IDA_datasets\Ash_Dieback\2021\GoPRO\ash_detection\all_images",unlabelled=r"\\gb010587mm\IDA_datasets\Ash_Dieback\2022\GoPRO\Unlabelled")
        #make_dataset_(all_data=all_data,df_county=df_county,df_info=df_info,under_represented=under_represented,max_postcodes = 60,dataset_size = 3000,outfolder=folder_previous[i+1])
        make_dataset_final(all_data=all_data,df_county=df_county,df_info=df_info,outfolder=folder_previous[i+1],max_postcodes = 1000,dataset_size = 5000)
        new_df=new_txtfile(older_dfinfo=df_info,output_folder=folder_previous[i+1])
        new_txtfile2(older_dfinfo=df_info,output_folder=folder_previous[i+1])
        count_df_new(new_df=new_df,output_folder=folder_previous[i+1])
        print(f"I am done with {i} cyclemate")
