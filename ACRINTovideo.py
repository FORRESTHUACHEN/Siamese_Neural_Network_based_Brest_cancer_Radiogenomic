# -*- coding: utf-8 -*-
"""
Created on Wed May 17 19:30:00 2023

@author: chenj
"""

import pylidc as pl
from pylidc.utils import volume_viewer
import matplotlib.pyplot as plt
import os
import pydicom
import SimpleITK as sitk
from pydicom import dcmread
import cv2
import numpy as np
import nibabel as nib
import pandas as pd
from skimage import transform,data
import scipy.ndimage.interpolation
from collections import Counter
from scipy import ndimage as ndi

SegmentationExcelFile=pd.read_csv('D:/DataSets/ACRIN-6698/DeepSegmentationMaskFiles/ACRINSegmentationResult.csv')

VideoOutputDir="D:/DataSets/ACRIN-6698/ACRINVideoForFeatureExtraction/"

for i in range(200,231):
    
    live_item=SegmentationExcelFile.iloc[i,0][-6:]
    print(str(live_item))
    # search_result=IndexColumn.isin([live_item])
    # index_in_original_raw=search_result.idxmax()
    live_item_raw_data=SegmentationExcelFile.iloc[i,3]
    
    live_item_path_class=live_item_raw_data.split('\\')
    live_item_father_node=""
    for dir_class in range(len(live_item_path_class)-2):
        live_item_father_node=live_item_father_node+'\\'+str(live_item_path_class[dir_class+1])
    live_item_father_node="D:\\DataSets\\ACRIN-6698\\ACRIN-6698"+ live_item_father_node
    
    live_item_father_class=os.listdir(live_item_father_node)
    live_item_box_mask=""
    for dir_class in range(len(live_item_father_class)):
        if "Mask" in live_item_father_class[dir_class]:
            live_item_box_mask=live_item_father_node+'\\'+str(live_item_father_class[dir_class])

    live_item_box_path=live_item_box_mask+'\\'+os.listdir(live_item_box_mask)[0]
    
    live_mask_ds=dcmread(live_item_box_path)
    live_mask_data=live_mask_ds.pixel_array
    live_mask_data = live_mask_data.transpose(1,2,0)
    start_slice=0
    end_slice=0
    upper_left_corner_x=0
    x_flag=np.zeros(live_mask_data.shape[1])
    mask_flag=np.zeros(live_mask_data.shape[2])
    y_flag=np.zeros(live_mask_data.shape[1])        
    for slices in range (live_mask_data.shape[2]):
        if 17 in live_mask_data[:,:,slices]:
            mask_flag[slices]=1
    start_slice=np.where(mask_flag==1)[0][0]+1
    end_slice=np.where(mask_flag==1)[0][-1]+1
    

    for columes in range(live_mask_data.shape[1]):
        if 17 in live_mask_data[:,columes,start_slice]:
            x_flag[columes]=1
        if 17 in live_mask_data[columes,:,start_slice]:
            y_flag[columes]=1
    upper_left_corner_y=np.where(x_flag==1)[0][0]+1
    upper_left_corner_x=np.where(y_flag==1)[0][0]+1
    
    if upper_left_corner_x<32:
        upper_left_corner_x=0
    elif upper_left_corner_x>160:
        upper_left_corner_x=128
    else:
        upper_left_corner_x=upper_left_corner_x-32
            
    if upper_left_corner_y<32:
        upper_left_corner_y=0
    elif upper_left_corner_y>160:
        upper_left_corner_y=128
    else:
        upper_left_corner_y=upper_left_corner_y-32

    
    live_item_raw_data_path=SegmentationExcelFile.iloc[i,3]
    reader = sitk.ImageSeriesReader()
    img_names = reader.GetGDCMSeriesFileNames(live_item_raw_data_path)   
    reader.SetFileNames(img_names)
    MRI_image = reader.Execute()
    MRI_image_array = sitk.GetArrayFromImage(MRI_image) # z, y, x
    MRI_image_array = MRI_image_array.transpose(1,2,0)
    
    MRI_image_array = MRI_image_array.astype(np.float64)
    if MRI_image_array.shape[0]!=256:
        MRI_image_array=scipy.ndimage.interpolation.zoom(MRI_image_array, [256.0/MRI_image_array.shape[0],256.0/MRI_image_array.shape[1],1.0], order=2, prefilter = True)
    window_center=np.mean(MRI_image_array)
    window_width=8*np.std(MRI_image_array)
    if window_center-(window_width/2)<0:
        window_high=window_center+(window_width/2)
    else:
        MRI_image_array=MRI_image_array-(window_center-window_width/2)
        window_high=window_width
    
    MRI_image_array[MRI_image_array>window_high]=window_high
    
    MRI_image_array_normalized=np.array(MRI_image_array/(window_high/255),dtype=np.uint8)
    
    Ph = np.uint8(MRI_image_array.shape[2]/live_mask_data.shape[2])

    video_array=np.zeros([128,128,3],dtype=np.uint8)
    
    slice_number_phase=live_mask_data.shape[2]
    for phases in range(Ph):
        
        video_name1=VideoOutputDir+'ISPY2-'+str(live_item)+'_phase_'+str(phases)+'.mp4'
        videowrite1 = cv2.VideoWriter(video_name1,-1,25,(128,128))#20是帧数，size是图片尺寸
        for available_slice in range(start_slice,end_slice):
            video_array[:,:,0]=MRI_image_array_normalized[upper_left_corner_x:upper_left_corner_x+128,upper_left_corner_y:upper_left_corner_y+128,slice_number_phase*phases+available_slice]
            video_array[:,:,1]=MRI_image_array_normalized[upper_left_corner_x:upper_left_corner_x+128,upper_left_corner_y:upper_left_corner_y+128,slice_number_phase*phases+available_slice]
            video_array[:,:,2]=MRI_image_array_normalized[upper_left_corner_x:upper_left_corner_x+128,upper_left_corner_y:upper_left_corner_y+128,slice_number_phase*phases+available_slice]
        
        
            
            for frame in range(25):
                videowrite1.write(video_array)
        
        videowrite1.release()
