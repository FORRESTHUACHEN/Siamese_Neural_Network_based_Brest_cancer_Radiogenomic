import matplotlib.pyplot as plt
import numpy as np
import mxnet as mx
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms
from gluoncv.data.transforms import video
from gluoncv import utils
from gluoncv.model_zoo import get_model
import scipy.ndimage

from gluoncv.utils.filesystem import try_import_decord
decord = try_import_decord()
import os

#video_root_fname = './videos2/'
#features_root_path='./features2/'

video_root_fname='D:/Datasets/RADCURE/VideosForFeatureExtraction/'
features_root_path='D:/Datasets/RADCURE/deepfeatures/'

video_name_list=os.listdir(video_root_fname)

model_name = 'i3d_inceptionv1_kinetics400'
net = get_model(model_name, nclass=400, pretrained=True)
print('%s model is successfully loaded.' % model_name)

for videoID in range(2700,2899):
    video_fname=video_root_fname+video_name_list[videoID]
    features_path=features_root_path+video_name_list[videoID][:-3]+'txt'
    vr = decord.VideoReader(video_fname)
    if len(vr)<150:
        intervals_number=5
    else:
        intervals_number=25
    frame_id_list = range(0, len(vr), intervals_number)
    video_data = vr.get_batch(frame_id_list).asnumpy()
    video_data_r=np.zeros((np.shape(video_data)[0],256,256,3))
    for i in range(np.shape(video_data)[0]):
        video_data_r[i,:,:,0]=scipy.ndimage.zoom(video_data[i,:,:,0],256/video_data.shape[1],order=1)
        video_data_r[i,:,:,1]=video_data_r[i,:,:,0]
        video_data_r[i,:,:,2]=video_data_r[i,:,:,0]
    clip_input = [video_data_r[vid, :, :, :] for vid, _ in enumerate(frame_id_list)]

    clip_input_num=np.shape(video_data)[0]

    transform_fn = video.VideoGroupValTransform(size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    clip_input = transform_fn(clip_input)
    clip_input = np.stack(clip_input, axis=0)
    clip_input = clip_input.reshape((-1,) + (clip_input_num, 3, 224, 224))
    clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))
    print(video_fname)
    pred = net(nd.array(clip_input))
    pred_vector=pred.asnumpy()
    f=open(features_path,'w')

    for i in range(400):
        print (pred_vector[0,i],file=f)
    f.close()







