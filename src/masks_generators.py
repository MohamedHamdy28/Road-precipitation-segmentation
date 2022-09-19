from dataclasses import dataclass
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import json
import numpy as np


def convert_to_colored():
    '''
        This function takes the annotation file in coco format and apply the labels on the images
    '''
    annotations_path = r'D:\University\intenships\Edge_vision_2022_internship\dataset\Evaluation dataset\all\not_mine_result.json'
    images = []
    with open(annotations_path) as handel:
        data = json.load(handel)
        for img in data['images']:
            images.append({'id':img['id'],'shape':(img['height'],img['width']),'file_name':img['file_name']})
        # for each imag_id there is a list with the poly of each class
        labels = {}
        for img in images:
            labels[img['id']] = [] # inside this dict will be list of all the annotations for this img id

        for ann in data['annotations']:
            labels[ann['image_id']].append(ann)
        classes = ['snow','wet','dry','other']
        categories = {
            '0':(129,96,49),
            '1':(141,21,239),
            '2':(0,145,225),
            '3':(255,55,0)
            }
        counter = 50
        for img in images:
            path = r'D:\University\intenships\Edge_vision_2022_internship\dataset\Evaluation dataset\all\not mine\\'+img['file_name'][19:]
            output_path = r'D:\University\intenships\Edge_vision_2022_internship\dataset\Evaluation dataset\all\not mine labeled colored\\'+str(counter)+'.png'
            numbered_path = r'D:\University\intenships\Edge_vision_2022_internship\dataset\Evaluation dataset\all\n\\'+str(counter)+'.png'
            blank = cv2.imread(path)
            m = cv2.imread(path)
            for label in labels[img['id']]:
                points = np.array(label['segmentation'],dtype=np.int32)
                points = points.reshape(-1,1,2)
                cv2.fillPoly(blank,pts=np.int32([points]),color=categories[str(label['category_id'])]) 
            blank = cv2.cvtColor(blank, cv2.COLOR_BGR2RGB)
            cv2.imwrite(output_path,blank)
            cv2.imwrite(numbered_path,m)
            counter +=1


def make_1d_target():
    '''
        This function is used to convert the mask or lable from 3 channels to 1 channel. For example, if we have
        RGB mask with the following dimension (256,256,3), this function will convert it to (256,256,1) where the 1 channel 
        will point to the class in which this pixel belongs 
    '''
    image_dir = r"D:\University\intenships\Edge_vision_2022_internship\dataset\Evaluation dataset\all\not mine labeled colored"
    images = os.listdir(image_dir)

    for img in images:
        img_path = image_dir+"\\"+img
        image = np.array(Image.open(img_path).convert("RGB"))
        new_image = np.zeros((image.shape[0],image.shape[1],1))
        output_path = r"D:\University\intenships\Edge_vision_2022_internship\dataset\Evaluation dataset\all\not mine 1d label\\"+img
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i][j][0]== 0 and image[i][j][1]== 145 and image[i][j][2]== 225:
                    new_image[i][j]=4 # meaning snow
                elif image[i][j][0]== 255 and image[i][j][1]== 55 and image[i][j][2]== 0:
                    new_image[i][j]=1 # wet
                elif image[i][j][0]== 129 and image[i][j][1]== 96 and image[i][j][2]== 49:
                    new_image[i][j]=2 # dry
                elif image[i][j][0]== 0 and image[i][j][1]== 0 and image[i][j][2]== 0:
                    new_image[i][j]=0 # background
                else:
                    new_image[i][j]=3 # other
        cv2.imwrite(output_path,new_image)
    

def main():
    convert_to_colored()
if __name__ == "__main__":
    main()




    
