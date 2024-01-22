# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 14:41:38 2023

@author: chenj
"""

import os 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from itertools import chain
import numpy as np
import pandas as pd


label=pd.read_excel('D:/DataSets/ACRIN-6698/Classifier-LSTM/ISPY2ClinicalLabel.xlsx')

Radiomics_feature=pd.read_csv('D:/DataSets/ACRIN-6698/Classifier-LSTM/ISPY2CoreSegmentationNormalizedRadiomicsSummary.csv')

Deep_feature=pd.read_csv('D:/DataSets/ACRIN-6698/Classifier-LSTM/ISPY2CoreDeepNormalizedFeatureSummary.csv')

Radiomics_feature_group=Radiomics_feature.groupby(['Patient_ID'])

Deep_feature_group=Deep_feature.groupby(['Patient_ID'])

Radiomics_label_unique=list(Radiomics_feature.loc[:,'Patient_ID'].unique())

# Deep_feature_unique=list(Deep_feature.loc[:,'Patient_ID'].unique())

# Common_samples=list(set(Radiomics_label_unique)&set(Deep_feature_unique))

writer1= tf.python_io.TFRecordWriter("D:/DataSets/ACRIN-6698/Classifier-LSTM/SubtypesDecodingCore_featureTrain2.tfrecords") 
writer2= tf.python_io.TFRecordWriter("D:/DataSets/ACRIN-6698/Classifier-LSTM/SubtypesDecodingCore_featureTest2.tfrecords")

 

index=list(range(380))
np.random.shuffle(index)

count1=0
count2=0

for i in range(len(Radiomics_label_unique)):
    Radiomics_feature_group_sample=Radiomics_feature_group.get_group(Radiomics_label_unique[i])
    Deep_feature_group_sample=Deep_feature_group.get_group(Radiomics_label_unique[i])
    
    Radiomics_feature_group_sample_data=np.array(Radiomics_feature_group_sample.iloc[:,2:],dtype=np.float32)
    Radiomics_feature_group_sample_data=Radiomics_feature_group_sample_data
    Deep_feature_group_sample_data=np.array(Deep_feature_group_sample.iloc[:,2:],dtype=np.float32)
    Deep_feature_group_sample_data=Deep_feature_group_sample_data
    if sum(label.Patient_ID==int(Radiomics_label_unique[i][-6:]))>0:
        
        Sample_label=label.loc[label.Patient_ID==int(Radiomics_label_unique[i][-6:]),:]
        
        Sample_label_value=int(Sample_label.iloc[0,1])
        
        if Sample_label_value>1:
            Sample_label_value=1
        Sample_label_value=np.array(Sample_label_value,dtype=np.int32)

        
        print(Sample_label_value)
        RealValidatedPhases=np.array(Deep_feature_group_sample_data.shape[0],dtype=np.int32)
    
        if Radiomics_feature_group_sample_data.shape[0]<9:
            for leftphase in range(9-Radiomics_feature_group_sample_data.shape[0]):
                # Radiomics_feature_group_sample_data=np.concatenate((Radiomics_feature_group_sample_data[0,:].reshape(1,-1),Radiomics_feature_group_sample_data),axis=0)
                # Deep_feature_group_sample_data=np.concatenate((Deep_feature_group_sample_data[0,:].reshape(1,-1),Deep_feature_group_sample_data),axis=0)
                Radiomics_feature_group_sample_data=np.concatenate((Radiomics_feature_group_sample_data,np.zeros([102,1],dtype=np.float32).reshape(1,-1)),axis=0)
                Deep_feature_group_sample_data=np.concatenate((Deep_feature_group_sample_data,np.zeros([400,1],dtype=np.float32).reshape(1,-1)),axis=0)
    
        Radiomics_feature_group_sample_data_towrite=Radiomics_feature_group_sample_data.tobytes()
        Deep_feature_group_sample_data_towrite=Deep_feature_group_sample_data.tobytes()
        RealValidatedPhases_towrite=RealValidatedPhases.tobytes()
        label_towrite=Sample_label_value.tobytes()
    
        example = tf.train.Example(features=tf.train.Features(feature={
           
            'RadiomicFeatures': tf.train.Feature(bytes_list=tf.train.BytesList(value=[Radiomics_feature_group_sample_data_towrite])),
            'DeepFeatures': tf.train.Feature(bytes_list=tf.train.BytesList(value=[Deep_feature_group_sample_data_towrite])),
            'ValidatedPhases':tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_towrite])),
            'Label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_towrite]))
            }))
    
        if i in index[0:310]:
            count1=count1+1
            writer1.write(example.SerializeToString())  
            print(Sample_label_value)
        else:
            count2=count2+1
            writer2.write(example.SerializeToString())  
        print(Sample_label_value)
writer1.close()
writer2.close()
