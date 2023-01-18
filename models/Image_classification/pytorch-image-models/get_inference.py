"""PyTorch Inference Script

An example inference script that outputs top-k class ids for images in a folder into a csv.

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import os
import time
import argparse
import logging
import numpy as np
import torch
import sys
import datetime

from timm.models import create_model, apply_test_time_pool
from timm.data import ImageDataset, create_loader, resolve_data_config
from timm.utils import AverageMeter, setup_default_logging

import tensorflow as tf
from tensorflow import keras

def health_class_Resnet50(model2classes_path,model4classes_path,data_folder):
    # first we need to load the models
    model2_clases = keras.models.load_model(model2classes_path)
    model4_clases = keras.models.load_model(model4classes_path)
    # image info
    image_size = (224, 224)
    # predictions
    health_classes = []
    img_name = []

    for item in os.listdir(data_folder):
        if item.endswith(".jpg") or item.endswith(".png"):
            img_name.append(item)
            test_image = keras.utils.load_img(os.path.join(data_folder,item), target_size=image_size)
            test_image = keras.utils.img_to_array(test_image)
            test_image = np.expand_dims(test_image,axis=0)

            # Decision model 1
            results_2classes = model2_clases.predict(test_image)
            # Decision model 2
            results_4classes = model4_clases.predict(test_image)
            # combinated probability
            if int(round(results_2classes[0][0],0)) == 1:
                comb_prob = [results_4classes[0][0]*(1-results_2classes[0][0]),results_4classes[0][1]*(1-results_2classes[0][0]),results_4classes[0][2]*results_2classes[0][0],results_4classes[0][3]*results_2classes[0][0]]
            
            else:
                comb_prob = [results_4classes[0][0]*(1-results_2classes[0][0]),results_4classes[0][1]*(1-results_2classes[0][0]),results_4classes[0][2]*results_2classes[0][0],results_4classes[0][3]*results_2classes[0][0]]
                
            health_classes.append(comb_prob.index(max(comb_prob)))

    return(health_classes)

def dieback_class_Resnet50(model2_clases,model4_clases,data_folder):
    # image info
    image_size = (224, 224)
    # predictions
    health_classes = []
    img_name = []

    for item in os.listdir(data_folder):
        if item.endswith(".jpg") or item.endswith(".png"):
            img_name.append(item)
            test_image = keras.utils.load_img(os.path.join(data_folder,item), target_size=image_size)
            test_image = keras.utils.img_to_array(test_image)
            test_image = np.expand_dims(test_image,axis=0)

            # Decision model 1
            results_2classes = model2_clases.predict(test_image)
            # Decision model 2
            results_4classes = model4_clases.predict(test_image)
            # combinated probability
            if int(round(results_2classes[0][0],0)) == 1:
                comb_prob = [results_4classes[0][0]*(1-results_2classes[0][0]),results_4classes[0][1]*(1-results_2classes[0][0]),results_4classes[0][2]*results_2classes[0][0],results_4classes[0][3]*results_2classes[0][0]]
            
            else:
                comb_prob = [results_4classes[0][0]*(1-results_2classes[0][0]),results_4classes[0][1]*(1-results_2classes[0][0]),results_4classes[0][2]*results_2classes[0][0],results_4classes[0][3]*results_2classes[0][0]]
                
            health_classes.append(comb_prob.index(max(comb_prob)))

    return(health_classes)

def dieback_batch_class(model2_clases,model4_clases,data_folder):
    # image info
    image_size = (224, 224)
    # predictions
    health_classes = []
    img_name = []
    minibatch = []
    # creating batches
    for item in os.listdir(data_folder):
        if item.endswith(".jpg") or item.endswith(".png"):
            if len(minibatch) < 32:
                # loading image
                test_image = keras.utils.load_img(os.path.join(data_folder,item), target_size=image_size)
                test_image = keras.utils.img_to_array(test_image)
                test_image = np.expand_dims(test_image,axis=0)
                minibatch.append(test_image)
            elif len(minibatch) == 32:
                img_name.append(minibatch)
                minibatch = []
    if len(minibatch) > 0:
        img_name.append(minibatch)
        
    # processing batches
    for batch in img_name:
        images = np.vstack(batch)

        # Decision model 1
        results_2classes = model2_clases.predict(images, batch_size = len(batch))
        # Decision model 2
        results_4classes = model4_clases.predict(images, batch_size = len(batch))

        # combined probability
        for ix in range(len(results_2classes)):
            if int(round(results_2classes[ix][0],0)) == 1:
                comb_prob = [results_4classes[ix][0]*(1-results_2classes[ix][0]),results_4classes[ix][1]*(1-results_2classes[ix][0]),results_4classes[ix][2]*results_2classes[ix][0],results_4classes[ix][3]*results_2classes[ix][0]]
            
            else:
                comb_prob = [results_4classes[ix][0]*(1-results_2classes[ix][0]),results_4classes[ix][1]*(1-results_2classes[ix][0]),results_4classes[ix][2]*results_2classes[ix][0],results_4classes[ix][3]*results_2classes[ix][0]]

            health_classes.append(comb_prob.index(max(comb_prob)))
        
    return(health_classes)

        # 
        # if int(round(results_2classes[0][0],0)) == 1:
        #     comb_prob = [results_4classes[0][0]*(1-results_2classes[0][0]),results_4classes[0][1]*(1-results_2classes[0][0]),results_4classes[0][2]*results_2classes[0][0],results_4classes[0][3]*results_2classes[0][0]]
            
        # else:
        #     comb_prob = [results_4classes[0][0]*(1-results_2classes[0][0]),results_4classes[0][1]*(1-results_2classes[0][0]),results_4classes[0][2]*results_2classes[0][0],results_4classes[0][3]*results_2classes[0][0]]
                
        # health_classes.append(comb_prob.index(max(comb_prob)))


def inference_classification(data_folder, pretrained=True,batch_size=2, model_weights='./weights/efficientnet_b2/model_best.pth.tar', model='efficientnet_b2',num_classes = 6,output_dir='./'):
    
    # initialising CuDA
    torch.backends.cudnn.benchmark = True
    _logger = logging.getLogger('inference')

    pretrained = pretrained or not model_weights

    # defining variables
    no_test_pool = False
    num_gpu = 1
    workers=0
    topk=1
    log_freq = 10
    # create model
    model = create_model(
        model,
        num_classes=num_classes,
        in_chans=3,
        pretrained=pretrained,
        checkpoint_path=model_weights)
    
    _logger.info('Model %s created, param count: %d' %
                 (model, sum([m.numel() for m in model.parameters()])))
    
    config = resolve_data_config({}, model=model)
    model, test_time_pool = (model, False) if no_test_pool else apply_test_time_pool(model, config)

    if num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(num_gpu))).cuda()
    else:
        model = model.cuda()
    
    loader = create_loader(
        ImageDataset(data_folder),
        input_size=config['input_size'],
        batch_size=batch_size,
        use_prefetcher=True,
        interpolation=config['interpolation'],
        mean=config['mean'],
        std=config['std'],
        num_workers=workers,
        crop_pct=1.0 if test_time_pool else config['crop_pct'])

    model.eval()

    k = min(topk, num_classes)
    batch_time = AverageMeter()
    end = time.time()
    topk_ids = []
    with torch.no_grad():
        for batch_idx, (input, _) in enumerate(loader):
            input = input.cuda()
            labels = model(input)
            topk = labels.topk(k)[1]
            topk_ids.append(topk.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % log_freq == 0:
                _logger.info('Predict: [{0}/{1}] Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                    batch_idx, len(loader), batch_time=batch_time))
    
    topk_ids = np.concatenate(topk_ids, axis=0)

    return(topk_ids)

    # with open(os.path.join(output_dir, './flightpath_classification.csv'), 'w') as out_file:
    #     filenames = loader.dataset.filenames(basename=True)
    #     for filename, label in zip(filenames, topk_ids):
    #         out_file.write('{0},{1}\n'.format(
    #             filename, ','.join([ str(v) for v in label])))
    
    # print("Csv file exported!")

def get_fp_assessment(data_folder,model_path=r"\\gb010587mm\Software_dev\Ash_Dieback_Solution\ash_dieback_solution\models\Image_classification\false_positives_model\logs\fine_tuning_final.keras"):
    fp_probs = []

    model = keras.models.load_model(model_path)
    image_size = (224, 224)
    batch_size = 1
        
    if os.path.isdir(data_folder):
        data_folder = data_folder
        for item in os.listdir(data_folder):
            if item.endswith(".jpg") or item.endswith(".png"):
                test_image = keras.utils.load_img(os.path.join(data_folder,item), target_size=image_size)
                test_image = keras.utils.img_to_array(test_image)
                test_image = np.expand_dims(test_image,axis=0)
                results = model.predict(test_image)
                fp_probs.append(results.item(0))
    elif isinstance(data_folder, np.ndarray):
        # resizing images
        if (data_folder.shape[1] == image_size[1]) and (data_folder.shape[0] == image_size[0]):
            print("No resizing needed")
            # expand dimensions
            test_image = np.expand_dims(data_folder,axis=0)
            
        else:
            resized_image = np.resize(data_folder,image_size)
            # expand dimensions
            test_image = np.expand_dims(resized_image,axis=0)
        # predicting
        results = model.predict(test_image)
        # appending predictions
        fp_probs.append(results.item(0))

    elif os.path.isfile(data_folder):
        test_image = keras.utils.load_img(data_folder, target_size=image_size)
        test_image = keras.utils.img_to_array(test_image)
        test_image = np.expand_dims(test_image,axis=0)
        results = model.predict(test_image)
        fp_probs.append(results.item(0))

    

    return(fp_probs)


if __name__ == "__main__":

    health_class1 = dieback_batch_class(model2_clases=r"\\gb010587mm\Software_dev\Ash_Dieback_Solution\Final_Model_weights\dieback_classification\Health_class_2.keras",model4_clases=r"\\gb010587mm\Software_dev\Ash_Dieback_Solution\Final_Model_weights\dieback_classification\Health_class_4.keras",data_folder=r"\\gb010587mm\Software_dev\Ash_Dieback_Solution\ash_dieback_solution\models\Image_classification\pytorch-image-models\Keras\SS\test")

    health_class_2 = health_class_Resnet50(model2classes_path=r"\\gb010587mm\Software_dev\Ash_Dieback_Solution\Final_Model_weights\dieback_classification\health_class_2.keras",model4classes_path=r"\\gb010587mm\Software_dev\Ash_Dieback_Solution\Final_Model_weights\dieback_classification\health_class_4.keras",data_folder=r"\\gb010587mm\Software_dev\Ash_Dieback_Solution\ash_dieback_solution\models\Image_classification\pytorch-image-models\Keras\SS\test")

    print("The new method: ")
    print(health_class1)
    print("The old method: ")
    print(health_class_2)
    
