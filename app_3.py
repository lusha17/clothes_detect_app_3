import copy
import sys
import gc
import os
from typing import List
from streamlit_webrtc import ClientSettings
from typing import List, NamedTuple, Optional
import cv2
import base64
import torch
import numpy as np
import pandas as pd
import streamlit as st
import pytz
import av
import datetime
import matplotlib.colors as mcolors
#from PIL import Image
import time
import threading
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, WebRtcMode, RTCConfiguration
from streamlit.legacy_caching import clear_cache
from aiortc.contrib.media import MediaPlayer

import click
import requests
from io import BytesIO
from pathlib import Path
import pickle
from PIL import Image as pil_img

from fastai.vision.data import ImageDataBunch
from fastai.vision.transform import get_transforms
from fastai.vision.learner import create_cnn
from fastai.vision import models
from fastai.vision.image import pil2tensor, Image
import matplotlib
import matplotlib.pyplot as plt
import cv2 as cv
import os
from imutils import resize
from lshash import LSHash
import numpy as np

lock = threading.Lock()

DEFAULT_CONFIDENCE_THRESHOLD = 0.35

CLASSES_CUSTOM = [ 'short sleeve top', 'long sleeve top','short sleeve outwear','long sleeve outwear','vest','sling','shorts','trousers','skirt','short sleeve dress', 'long sleeve dress','vest dress','sling dress']

def main():
    gc.enable()
    st.header("Fashion Items Detection Demo")
    st.sidebar.markdown("""<center data-parsed=""><img src="http://drive.google.com/uc?export=view&id=1D-pN81CupHMcxb7xa5-Z6JZZIbagRqH_" align="center"></center>""",unsafe_allow_html=True,)
    st.sidebar.markdown(" ")
    
    def reload():
        filelist = [ f for f in os.listdir('data')]
        for f in filelist:
            os.remove(os.path.join('data', f))
        clear_cache()
        gc.collect()
        st.experimental_rerun()
        
    def get_yolo5():
        return torch.hub.load('ultralytics/yolov5', 'custom', path='last_s.pt')
        
    with st.spinner('Loading the model...'):
        cache_key = 'custom'
        if cache_key in st.session_state:
            model = st.session_state[cache_key]
        else:
            model = get_yolo5()
            st.session_state[cache_key] = model
            
    pages = st.sidebar.columns([1, 1, 1])
    pages[0].markdown(" ")
    
    if pages[1].button("Reload App"):
        reload()
    prediction_mode = st.sidebar.radio("", ('Single image', 'Web camera', 'Local video', 'Detect SKU in image'), index=1)
    
    if prediction_mode == 'Single image':
        func_image(model)
    elif prediction_mode == 'Web camera':
        func_web(model)
    elif prediction_mode == 'Local video':
        func_video(model)
    elif prediction_mode == 'Detect SKU in image':
        func_detect_sku(model)
        
def func_image(model):
    RTC_CONFIGURATION = RTCConfiguration(
       {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ) 

    CLASSES_CUSTOM = [ 'short sleeve top', 'long sleeve top','short sleeve outwear','long sleeve outwear','vest','sling','shorts','trousers','skirt','short sleeve dress', 'long sleeve dress','vest dress','sling dress']
    

    DEFAULT_CONFIDENCE_THRESHOLD = 0.35

    result_queue = [] 


    def get_preds(img):
        return model([img]).xyxy[0].numpy()

    def get_colors(indexes):
        to_255 = lambda c: int(c*255)
        tab_colors = list(mcolors.TABLEAU_COLORS.values())
        tab_colors = [list(map(to_255, mcolors.to_rgb(name_color))) for name_color in tab_colors]
        base_colors = list(mcolors.BASE_COLORS.values())
        base_colors = [list(map(to_255, name_color)) for name_color in base_colors]
        rgb_colors = tab_colors + base_colors
        rgb_colors = rgb_colors*5
        color_dict = {}
        for i, index in enumerate(indexes):
            if i < len(rgb_colors):
                color_dict[index] = rgb_colors[i]
            else:
                color_dict[index] = (255,0,0)
        return color_dict
    
    def create_player():
        return MediaPlayer(path)

    

    
    
            
    confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05)

    #prediction_mode = st.sidebar.radio("", ('Single image', 'Web camera', 'Local video'), index=2)
    CLASSES = CLASSES_CUSTOM
    classes_selector = st.sidebar.multiselect('Select classes', CLASSES, default='short sleeve top')

    all_labels_chbox = st.sidebar.checkbox('All classes', value=True)
    if all_labels_chbox:
        target_class_ids = list(range(len(CLASSES)))
    elif classes_selector:
        target_class_ids = [CLASSES.index(class_name) for class_name in classes_selector]
    else:
        target_class_ids = [0]
    rgb_colors = get_colors(target_class_ids)
    detected_ids = None
    uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = get_preds(img)
        result_copy = result.copy()
        result_copy = result_copy[np.isin(result_copy[:,-1], target_class_ids)]
        detected_ids = []
        img_draw = img.copy().astype(np.uint8)
        for bbox_data in result_copy:
            xmin, ymin, xmax, ymax, conf, label = bbox_data
            if conf > confidence_threshold:
                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)
                p0, p1, label = (xmin, ymin), (xmax, ymax), int(label)
                img_draw = cv2.rectangle(img_draw, p0, p1, rgb_colors[label], 2) 
                ytext = ymin - 10 if ymin - 10 > 10 else ymin + 15
                xtext = xmin + 10
                class_ = CLASSES[label]
                text_for_vis = '{} {}'.format(class_, str(conf.round(2)))
                img_draw = cv2.putText(img_draw, text_for_vis, (int(xtext), int(ytext)),cv2.FONT_HERSHEY_SIMPLEX,0.5,rgb_colors[label],2,)
                detected_ids.append(label)
        st.image(img_draw, use_column_width=True)
    
    
def func_web(model):
    RTC_CONFIGURATION = RTCConfiguration(
       {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ) 

    CLASSES_CUSTOM = [ 'short sleeve top', 'long sleeve top','short sleeve outwear','long sleeve outwear','vest','sling','shorts','trousers','skirt','short sleeve dress', 'long sleeve dress','vest dress','sling dress']
    

    DEFAULT_CONFIDENCE_THRESHOLD = 0.35

    result_queue = [] 


    def get_preds(img):
        return model([img]).xyxy[0].numpy()

    def get_colors(indexes):
        to_255 = lambda c: int(c*255)
        tab_colors = list(mcolors.TABLEAU_COLORS.values())
        tab_colors = [list(map(to_255, mcolors.to_rgb(name_color))) for name_color in tab_colors]
        base_colors = list(mcolors.BASE_COLORS.values())
        base_colors = [list(map(to_255, name_color)) for name_color in base_colors]
        rgb_colors = tab_colors + base_colors
        rgb_colors = rgb_colors*5
        color_dict = {}
        for i, index in enumerate(indexes):
            if i < len(rgb_colors):
                color_dict[index] = rgb_colors[i]
            else:
                color_dict[index] = (255,0,0)
        return color_dict
    
    def create_player():
        return MediaPlayer(path)


    def transform(frame):
        img = frame.to_ndarray(format="bgr24")        
        img_ch = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = get_preds(img_ch)
        result = result[np.isin(result[:,-1], target_class_ids)]  
        for bbox_data in result:
            xmin, ymin, xmax, ymax, conf, label = bbox_data
            if conf > confidence_threshold:
                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)
                p0, p1, label = (xmin, ymin), (xmax, ymax), int(label)
                class_ = CLASSES[label]
                img = cv2.rectangle(img, p0, p1, rgb_colors[label], 2) 
                ytext = ymin - 10 if ymin - 10 > 10 else ymin + 15
                xtext = xmin + 10
                text_for_vis = '{} {}'.format(class_, str(conf.round(2)))
                img = cv2.putText(img, text_for_vis, (int(xtext), int(ytext)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb_colors[label], 2,)
                if agree:
                    time_detect = datetime.datetime.now(pytz.timezone("America/New_York")).replace(tzinfo=None).strftime("%m-%d-%y %H:%M:%S")
                    cropped_image = img[ymin:ymax, xmin:xmax]
                    retval, buffer_img= cv2.imencode('.jpg', cropped_image)
                    data = base64.b64encode(buffer_img).decode("utf-8")
                    html = "<img src='data:image/jpg;base64," + data + f"""' style='display:block;margin-left:auto;margin-right:auto;width:200px;border:0;'>"""
                    with lock:
                        result_queue.insert(0, {'object': class_, 'time_detect': time_detect, 'confident': str(conf.round(2)), 'img': html})
                img = cv2.rectangle(img, p0, p1, rgb_colors[label], 2) 
                ytext = ymin - 10 if ymin - 10 > 10 else ymin + 15
                xtext = xmin + 10
                text_for_vis = '{} {}'.format(class_, str(conf.round(2)))
                img = cv2.putText(img, text_for_vis, (int(xtext), int(ytext)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb_colors[label], 2,)
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
            
    confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05)

    #prediction_mode = st.sidebar.radio("", ('Single image', 'Web camera', 'Local video'), index=2)
    CLASSES = CLASSES_CUSTOM
    classes_selector = st.sidebar.multiselect('Select classes', CLASSES, default='short sleeve top')

    all_labels_chbox = st.sidebar.checkbox('All classes', value=True)
    if all_labels_chbox:
        target_class_ids = list(range(len(CLASSES)))
    elif classes_selector:
        target_class_ids = [CLASSES.index(class_name) for class_name in classes_selector]
    else:
        target_class_ids = [0]
    rgb_colors = get_colors(target_class_ids)
    detected_ids = None
    ctx = webrtc_streamer(key="example", video_frame_callback=transform,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False}, mode=WebRtcMode.SENDRECV, async_processing=True)
    agree = st.checkbox("Enable clothes logging", value=False)
    if agree:
        if ctx.state.playing:
            labels_placeholder = st.empty()
            while True:
                time.sleep(0.5)
                with lock:
                    result_queue = result_queue[:5]
                    df = pd.DataFrame(result_queue)
                    labels_placeholder.write(df.to_html(escape=False), unsafe_allow_html=True)
    
def func_video(model):
    def load_image_databunch(input_path, classes):
        """
        Code to define a databunch compatible with model
        """
        tfms = get_transforms(
            do_flip=False,
            flip_vert=False,
            max_rotate=0,
            max_lighting=0,
            max_zoom=1,
            max_warp=0,
        )

        data_bunch = ImageDataBunch.single_from_classes(
            Path(input_path), classes, ds_tfms=tfms, size=640
        )

        return data_bunch
    
    def load_model(data_bunch, model_type, model_name):
        """
        Function to create and load pretrained weights of convolutional learner
        """
        learn = create_cnn(data_bunch, model_type, pretrained=False)
        learn.load(model_name)
        return learn


    class SaveFeatures:
        features = None

        def __init__(self, m):
            self.hook = m.register_forward_hook(self.hook_fn)
            self.features = None

        def hook_fn(self, module, input, output):
            out = output.detach().cpu().numpy()
            if isinstance(self.features, type(None)):
                self.features = out
            else:
                self.features = np.row_stack((self.features, out))

        def remove(self):
            self.hook.remove()
    
    def image_to_vec(url_img, hook, learner):
        _ = learner.predict(Image(pil2tensor(url_img, np.float32).div_(255)))
        vect = hook.features[-1]
        return vect

    def get_vect(url_img, conv_learn, hook):
        vect = image_to_vec(url_img, hook, conv_learn)
        return vect
    
    def get_list_similar_images(url_img, conv_learn, hook, lsh, output_path, n_items):
        ## Converting Image to vector
        vect = image_to_vec(url_img, hook, conv_learn)
        ## print(vect)
        ## Finding approximate nearest neighbours using LSH
        response = lsh.query(vect, num_results=n_items+1, distance_func="hamming")
        list_images = []
        for res in response:
            list_images.append(res[0][1])
        return list_images
    
    def get_most_similar_image(img1, list_similar_images, w , h):
        #img1 = cv.imread('cropped_image.jpg',cv.IMREAD_GRAYSCALE)
        img1 = cv2.cvtColor(img1, cv.COLOR_RGB2GRAY)
        img1 = resize(img1,w, h)
        bool_find = False
        for path_similar_image in list_similar_images:
            img2 = cv.imread('cropped_dataset/'+path_similar_image,cv.IMREAD_GRAYSCALE) 
            good_matches1, keypoints11, keypoints21 = get_key_points_by_images(img1, img2)
            good_matches2, keypoints12, keypoints22 = get_key_points_by_images(img2, img1)
            if (len(good_matches1) > 40) | (len(good_matches2) > 40):
                bool_find = True
                break
        if bool_find:
            img3 = cv.imread('cropped_dataset/'+path_similar_image,cv.IMREAD_COLOR) 
        else:
            img3 = None
        return bool_find, img3
    
    
    def detect_sku(im, cropped_image_cv):
        w, h = im.size
        resized_hight = 540
        wpercent = (resized_hight/float(h))
        resized_w = int((float(w)*float(wpercent)))
        im = im.resize((resized_w,resized_hight), pil_img.ANTIALIAS)
        im = im.convert('RGB')
        list_similar_images = get_list_similar_images(im, learner, sf, lsh, "output/output.png", 4)
        bool_find, most_similar_image = get_most_similar_image(cropped_image_cv, list_similar_images, resized_w , resized_hight)
        return bool_find, most_similar_image
            
    RTC_CONFIGURATION = RTCConfiguration(
       {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ) 

    CLASSES_CUSTOM = [ 'short sleeve top', 'long sleeve top','short sleeve outwear','long sleeve outwear','vest','sling','shorts','trousers','skirt','short sleeve dress', 'long sleeve dress','vest dress','sling dress']
    

    DEFAULT_CONFIDENCE_THRESHOLD = 0.35

    result_queue = [] 
    frames_count = 0
    
    classes = [ 'short sleeve top', 'long sleeve top','short sleeve outwear','long sleeve outwear','sling','shorts','trousers','skirt','short sleeve dress', 'long sleeve dress', 'vest dress','sling dress']
    data_bunch = load_image_databunch("cropped_dataset", classes)
    learner = load_model(data_bunch, models.resnet34, "stg1-rn34")
    sf = SaveFeatures(learner.model[1][5])
    lsh = pickle.load(open("lsh.p", "rb"))
    
    detector = cv.BRISK_create()
    descriptor = cv.SIFT_create()
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    FLANN = cv.FlannBasedMatcher(indexParams = index_params, searchParams = search_params)
    ratio_thresh_for_key_points = 0.8
    
    def get_key_points_by_images(image1, image2):
        detector = cv.BRISK_create()
        descriptor = cv.SIFT_create()
        keypoints1, descriptors1 = features(image1, detector, descriptor)
        keypoints2, descriptors2 = features(image2, detector, descriptor)
        good_matches = matcher(image1, image2, keypoints1, keypoints2, descriptors1, descriptors2)
        return good_matches, keypoints1, keypoints2

    def matcher(image1, image2, keypoints1, keypoints2, descriptors1, descriptors2):
        descriptors1 = np.float32(descriptors1)
        descriptors2 = np.float32(descriptors2)
        matches = FLANN.knnMatch(queryDescriptors = descriptors1, trainDescriptors = descriptors2, k = 2)
        good_matches = []
        for m, n in matches:
                if m.distance < ratio_thresh_for_key_points * n.distance:
                    good_matches.append(m)
        return good_matches

    def features(image, detector, descriptor):
        keypoints = detector.detect(image, None)
        keypoints, descriptors = descriptor.compute(image, keypoints)
        return keypoints, descriptors
    
    
    def img_to_html(img):
        retval, buffer_img= cv2.imencode('.jpg', img)
        data = base64.b64encode(buffer_img).decode("utf-8")
        html = "<img src='data:image/jpg;base64," + data + f"""' style='display:block;margin-left:auto;margin-right:auto;width:200px;border:0;'>"""
        return html

    def get_preds(img):
        return model([img]).xyxy[0].numpy()

    def get_colors(indexes):
        to_255 = lambda c: int(c*255)
        tab_colors = list(mcolors.TABLEAU_COLORS.values())
        tab_colors = [list(map(to_255, mcolors.to_rgb(name_color))) for name_color in tab_colors]
        base_colors = list(mcolors.BASE_COLORS.values())
        base_colors = [list(map(to_255, name_color)) for name_color in base_colors]
        rgb_colors = tab_colors + base_colors
        rgb_colors = rgb_colors*5
        color_dict = {}
        for i, index in enumerate(indexes):
            if i < len(rgb_colors):
                color_dict[index] = rgb_colors[i]
            else:
                color_dict[index] = (255,0,0)
        return color_dict
    
    
    def create_player():
        return MediaPlayer(path)
    

    def transform(frame):
        #img = frame.to_ndarray(format="bgr24")   
        img = frame
        img_ch = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = get_preds(img_ch)
        result = result[np.isin(result[:,-1], target_class_ids)]  
        dict_cropped_image = {}
        for bbox_data in result:
            xmin, ymin, xmax, ymax, conf, label = bbox_data
            conf = round(conf, 2)
            if conf >= confidence_threshold:
                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)
                p0, p1, label = (xmin, ymin), (xmax, ymax), int(label)
                class_ = CLASSES[label]
                cropped_image = img[ymin:ymax, xmin:xmax]
                w, h, t = cropped_image.shape
                dict_cropped_image[w+h] = cropped_image
        if len(dict_cropped_image)!=0:
            list_els = list(dict_cropped_image.keys())
            list_els.sort()
            list_els = list_els[-3:]
            #print(list_els)
            for el in list_els:
                cropped_image_cv = dict_cropped_image[el]
                cropped_image_pil = pil_img.fromarray(cropped_image_cv)
                bool_find, sku_image = detect_sku(cropped_image_pil, cropped_image_cv)
                html_cropped_image_cv = img_to_html(cropped_image_cv)
                if bool_find:
                    html_sku_image = img_to_html(sku_image)
                    result_queue.insert(0, {'cropped_frame': html_cropped_image_cv, 'sku': html_sku_image})
                else:
                    html_sku_image = 'No match found'
        #return av.VideoFrame.from_ndarray(img, format="bgr24")
        return img
    
    
            
    confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05)

    #prediction_mode = st.sidebar.radio("", ('Single image', 'Web camera', 'Local video'), index=2)
    CLASSES = CLASSES_CUSTOM
    classes_selector = st.sidebar.multiselect('Select classes', CLASSES, default='short sleeve top')

    all_labels_chbox = st.sidebar.checkbox('All classes', value=True)
    if all_labels_chbox:
        target_class_ids = list(range(len(CLASSES)))
    elif classes_selector:
        target_class_ids = [CLASSES.index(class_name) for class_name in classes_selector]
    else:
        target_class_ids = [0]
    rgb_colors = get_colors(target_class_ids)
    detected_ids = None
    if 'path' not in locals():
        path = ''
    uploaded_video = st.file_uploader("Upload video", type = ['mp4','mpeg','mov'], accept_multiple_files=False)
    if uploaded_video != None:
        vid = uploaded_video.name
        last_name = path.replace("data/",'')
        if last_name!=vid:
            #print('last_name!=vid')
            #print(last_name)
            #print(vid)
            filelist = [f for f in os.listdir('data')]
            for f in filelist:
                os.remove('data/'+f)
        path = 'data/' + vid
        with open(path, mode='wb') as f:
            f.write(uploaded_video.read())
        #path = 'test.mp4'
        if st.button('Processing video'):
            video = cv2.VideoCapture(path)
            while True:
                frames_count+=1
                #print(frames_count)
                ret, frame = video.read()
                #print(result_queue)
                if ret == True:
                    if frames_count%20!=0:
                        continue
                    img_res = transform(frame)
                else:
                    break    
            video.release()
    labels_placeholder = st.empty()
    #agree = st.checkbox("Enable clothes logging", value=False)
    df = pd.DataFrame(result_queue)
    labels_placeholder.write(df.to_html(escape=False), unsafe_allow_html=True)
                    
def func_detect_sku(model):
    def load_image_databunch(input_path, classes):
        """
        Code to define a databunch compatible with model
        """
        tfms = get_transforms(
            do_flip=False,
            flip_vert=False,
            max_rotate=0,
            max_lighting=0,
            max_zoom=1,
            max_warp=0,
        )

        data_bunch = ImageDataBunch.single_from_classes(
            Path(input_path), classes, ds_tfms=tfms, size=640
        )

        return data_bunch

    def load_model(data_bunch, model_type, model_name):
        """
        Function to create and load pretrained weights of convolutional learner
        """
        learn = create_cnn(data_bunch, model_type, pretrained=False)
        learn.load(model_name)
        return learn


    class SaveFeatures:
        features = None

        def __init__(self, m):
            self.hook = m.register_forward_hook(self.hook_fn)
            self.features = None

        def hook_fn(self, module, input, output):
            out = output.detach().cpu().numpy()
            if isinstance(self.features, type(None)):
                self.features = out
            else:
                self.features = np.row_stack((self.features, out))

        def remove(self):
            self.hook.remove()

    def image_to_vec(url_img, hook, learner):
        '''
        Function to convert image to vector
        '''
        _ = learner.predict(Image(pil2tensor(url_img, np.float32).div_(255)))
        vect = hook.features[-1]
        return vect

    def get_vect(url_img, conv_learn, hook):
        ## Converting Image to vector
        vect = image_to_vec(url_img, hook, conv_learn)
        return vect

    def detect_sku(im, cropped_image_cv):
        classes = [ 'short sleeve top', 'long sleeve top','short sleeve outwear','long sleeve outwear','sling','shorts','trousers','skirt','short sleeve dress', 'long sleeve dress', 'vest dress','sling dress']
        data_bunch = load_image_databunch("cropped_dataset", classes)
        learner = load_model(data_bunch, models.resnet34, "stg1-rn34")
        sf = SaveFeatures(learner.model[1][5])
        w, h = im.size
        resized_hight = 540
        wpercent = (resized_hight/float(h))
        resized_w = int((float(w)*float(wpercent)))
        #print(resized_w,resized_hight)
        im = im.resize((resized_w,resized_hight), pil_img.ANTIALIAS)
        im = im.convert('RGB')
        lsh = pickle.load(open("lsh.p", "rb"))
        list_similar_images, img_res_1 = view_similar_images(im, learner, sf, lsh, "output/output.png", 5)
        #img_res_1 = cv.imread('output/output.png', 1)
        st.pyplot(img_res_1)
        bool_find, img_res_2 = get_most_similar_image(cropped_image_cv, list_similar_images, resized_w , resized_hight)
        if bool_find:
            st.image(img_res_2)


    def get_key_points_by_images(image1, image2):
        detector = cv.BRISK_create()
        descriptor = cv.SIFT_create()
        keypoints1, descriptors1 = features(image1, detector, descriptor)
        keypoints2, descriptors2 = features(image2, detector, descriptor)
        good_matches = matcher(image1, image2, keypoints1, keypoints2, descriptors1, descriptors2)
        return good_matches, keypoints1, keypoints2

    def matcher(image1, image2, keypoints1, keypoints2, descriptors1, descriptors2):
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        descriptors1 = np.float32(descriptors1)
        descriptors2 = np.float32(descriptors2)
        FLANN = cv.FlannBasedMatcher(indexParams = index_params, searchParams = search_params)
        matches = FLANN.knnMatch(queryDescriptors = descriptors1, trainDescriptors = descriptors2, k = 2)
        ratio_thresh = 0.8
        good_matches = []
        for m, n in matches:
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)
        return good_matches

    def features(image, detector, descriptor):
        keypoints = detector.detect(image, None)
        keypoints, descriptors = descriptor.compute(image, keypoints)
        return keypoints, descriptors


    def get_most_similar_image(img1, list_similar_images, w , h):
        #img1 = cv.imread('cropped_image.jpg',cv.IMREAD_GRAYSCALE)
        img1 = cv2.cvtColor(img1, cv.COLOR_RGB2GRAY)
        img1 = resize(img1,w, h)
        bool_find = False
        for path_similar_image in list_similar_images:
            img2 = cv.imread('cropped_dataset/'+path_similar_image,cv.IMREAD_GRAYSCALE) 
            good_matches1, keypoints11, keypoints21 = get_key_points_by_images(img1, img2)
            good_matches2, keypoints12, keypoints22 = get_key_points_by_images(img2, img1)
            if (len(good_matches1) > 40) | (len(good_matches2) > 40):
                bool_find = True
                break
        if bool_find:
            img3 =  cv.drawMatches(img1, keypoints11, img2, keypoints21, good_matches1, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            #img3 = cv.drawMatchesKnn(img1,keypoints11,img2,keypoints21,good_matches1,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        else:
            img3 = None
        return bool_find, img3


    def view_similar_images(url_img, conv_learn, hook, lsh, output_path, n_items):
        ## Converting Image to vector
        vect = image_to_vec(url_img, hook, conv_learn)
        #print(vect)
        ## Finding approximate nearest neighbours using LSH
        response = lsh.query(vect, num_results=n_items+1, distance_func="hamming")
        list_images = []
        for res in response:
            list_images.append(res[0][1])
        ## Dimension calculation for plotting
        columns = 3
        rows = int(np.ceil(n_items + 1 / columns)) + 1
        ## Plotting function
        fig = plt.figure(figsize=(2 * rows, 3 * rows))
        for i in range(0, columns * rows + 1):
            if i == 0:
                fig.add_subplot(rows, columns, i + 1)
                plt.imshow(url_img)
                plt.axis("off")
                plt.title("Input Image")
            elif i < n_items + 2:
                ret_img = pil_img.open('cropped_dataset/' + response[i - 1][0][1])
                fig.add_subplot(rows, columns, i + 2)
                plt.imshow(ret_img)
                plt.axis("off")
                plt.title(str(i - 1))
        fig.tight_layout()
        #fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
        #img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        #img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return list_images, fig

    CLASSES_CUSTOM = [ 'short sleeve top', 'long sleeve top','short sleeve outwear','long sleeve outwear','vest','sling','shorts','trousers','skirt','short sleeve dress', 'long sleeve dress','vest dress','sling dress']
    DEFAULT_CONFIDENCE_THRESHOLD = 0.35

    result_queue = [] 


    def get_preds(img):
        return model([img]).xyxy[0].numpy()

    def get_colors(indexes):
        to_255 = lambda c: int(c*255)
        tab_colors = list(mcolors.TABLEAU_COLORS.values())
        tab_colors = [list(map(to_255, mcolors.to_rgb(name_color))) for name_color in tab_colors]
        base_colors = list(mcolors.BASE_COLORS.values())
        base_colors = [list(map(to_255, name_color)) for name_color in base_colors]
        rgb_colors = tab_colors + base_colors
        rgb_colors = rgb_colors*5
        color_dict = {}
        for i, index in enumerate(indexes):
            if i < len(rgb_colors):
                color_dict[index] = rgb_colors[i]
            else:
                color_dict[index] = (255,0,0)
        return color_dict
    
    confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05)

    CLASSES = CLASSES_CUSTOM
    classes_selector = st.sidebar.multiselect('Select classes', CLASSES, default='short sleeve top')

    all_labels_chbox = st.sidebar.checkbox('All classes', value=True)
    if all_labels_chbox:
        target_class_ids = list(range(len(CLASSES)))
    elif classes_selector:
        target_class_ids = [CLASSES.index(class_name) for class_name in classes_selector]
    else:
        target_class_ids = [0]
    rgb_colors = get_colors(target_class_ids)
    detected_ids = None
    uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
        img_orig = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        result = get_preds(img)
        result_copy = result.copy()
        result_copy = result_copy[np.isin(result_copy[:,-1], target_class_ids)]
        detected_ids = []
        img_draw = img.copy().astype(np.uint8)
        dict_cropped_image = {}
        for bbox_data in result_copy:
            xmin, ymin, xmax, ymax, conf, label = bbox_data
            conf = round(conf, 2)
            if conf >= confidence_threshold:
                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)
                cropped_image = img[ymin:ymax, xmin:xmax]
                w, h, t = cropped_image.shape
                dict_cropped_image[w+h]=cropped_image
        if len(dict_cropped_image)!=0:
            max_element = max(dict_cropped_image.keys())
            cropped_image_cv = dict_cropped_image[max_element]
            cropped_image_pil = pil_img.fromarray(cropped_image_cv)
            #cv.imwrite('cropped_image.jpg', cropped_image)
            #cropped_image = pil_img.open('cropped_image.jpg')
            #image1 = cv.imread(filename = 'cropped_image.jpg', flags = cv.IMREAD_GRAYSCALE)
            detect_sku(cropped_image_pil, cropped_image_cv)
            
            

if __name__ == "__main__":
    main()