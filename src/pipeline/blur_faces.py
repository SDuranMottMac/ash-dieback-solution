'''
This component aims to blurry faces of people in images
'''

import os
import argparse
import cv2
from DetectorAPI import Detector
import matplotlib.pyplot as plt



def blurBoxes(image, boxes):
    """
    Argument:
    image -- the image that will be edited as a matrix
    boxes -- list of boxes that will be blurred each element must be a dictionary that has [id, score, x1, y1, x2, y2] keys

    Returns:
    image -- the blurred image as a matrix
    """

    for box in boxes:
        # unpack each box
        x1, y1 = box["x1"], box["y1"]
        x2, y2 = box["x2"], box["y2"]

        # crop the image due to the current box
        sub = image[y1:y2, x1:x2]

        # apply GaussianBlur on cropped area
        blur = cv2.blur(sub, (25, 25))

        # paste blurred image on the original image
        image[y1:y2, x1:x2] = blur

    return image


def blur_faces(input_folder,model,threshold=0.4):
    # initialize the detector

    model_path = model
    threshold = threshold
    detector = Detector(model_path = model_path, name="detection")

    # looping over the images
    for item in os.listdir(input_folder):
        print("The item to be processed is: "+str(item))
        if item.endswith(".db"):
            continue
        # load the image
        image = cv2.imread(os.path.join(input_folder, item))

        # detect faces
        faces = detector.detect_objects(image, threshold=threshold)

        # apply burring
        image = blurBoxes(image, faces)

        # save the image
        cv2.imwrite(os.path.join(input_folder, item), image)

def detect_license(img):
    #Load the license cascade
    license_cascade = cv2.CascadeClassifier('./weights/blur_plates/haarcascade_russian_plate_number.xml')

    #Store copy of the image in color and B&W to manipulate
    img_copy = img.copy()
    img_BW = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    #Returns coordinates of the located license plate
    license_rect = license_cascade.detectMultiScale(img_BW, scaleFactor=1.3, minNeighbors=5)

    #Isolate ROI, blur and reapply onto original image
    for (x,y,w,h) in license_rect:
        license_ROI = img_copy[y:y+h, x:x+w]
        license_ROI = cv2.blur(license_ROI,(30,30))
        img_copy[y:y+h, x:x+w] = license_ROI
    
    return img_copy  

def detecting_plate_number(input_folder):
    # looping over the images
    for item in os.listdir(input_folder):
        # load the image
        image = cv2.imread(os.path.join(input_folder, item))

        # detect faces
        result_blur = detect_license(image)
        result_blur = cv2.cvtColor(result_blur, cv2.COLOR_BGR2RGB)

        # save the image
        cv2.imwrite(os.path.join(input_folder, item), result_blur)

if __name__ == "__main__":

    img_directory = r"\\gb010587mm\Software_dev\Ash_Dieback_Solution\Tests\Blurring_images"
    blur_faces(input_folder=img_directory,model='./weights/blur_faces/face.pb',threshold=0.15)