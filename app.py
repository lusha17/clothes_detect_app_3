import sys
import gc
import os
from imutils import resize
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
from PIL import Image
import time
import threading
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, WebRtcMode, RTCConfiguration
from streamlit.legacy_caching import clear_cache
from aiortc.contrib.media import MediaPlayer

st.set_page_config(layout="wide")
from streamlit.scriptrunner import get_script_run_ctx as get_report_ctx




lock = threading.Lock()

hide_table_row_index = """<style>
                thead tr th:first-child {display:none}
                tbody th {display:none}
                </style>
                """

DEFAULT_CONFIDENCE_THRESHOLD = 0.3

CLASSES_CUSTOM = [ 'short sleeve top', 'long sleeve top', 'short sleeve outwear', 'long sleeve outwear',
                  'vest', 'sling', 'shorts', 'trousers', 'skirt', 'short sleeve dress', 'long sleeve dress',
                  'vest dress', 'sling dress', 'dress', 'handbag', 'swimwear']

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
        
    pages = st.sidebar.columns([1, 1, 1])
    pages[0].markdown(" ")
    
    if pages[1].button("Reload App"):
        reload()
        
    def get_yolo5():
        return torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
        
    with st.spinner('Loading the model...'):
        cache_key = 'custom'
        if cache_key in st.session_state:
            model = st.session_state[cache_key]
        else:
            model = get_yolo5()
            st.session_state[cache_key] = model
            
    prediction_mode = st.sidebar.radio("", ('Single image', 'Web camera', 'Local video'), index=1)
    if prediction_mode == 'Single image':
        func_image(model)
    elif prediction_mode == 'Web camera':
        func_web(model)
    elif prediction_mode == 'Local video':
        func_video(model)
        
def func_image(model):
    
    RTC_CONFIGURATION = RTCConfiguration(
       {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ) 

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
    #ctx1 = get_report_ctx()
    #print(ctx1.session_id)
    result_queue = []
    
    global frames_count1
    frames_count1 = 0


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
        with lock:
            global frames_count1
            frames_count1+=1
        
        #print(frames_count)
        if frames_count1%4!=0:
            return frame
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
                img = cv2.rectangle(img, p0, p1, rgb_colors[label], 2) 
                ytext = ymin - 10 if ymin - 10 > 10 else ymin + 15
                xtext = xmin + 10
                class_ = CLASSES[label]
                text_for_vis = '{} {}'.format(class_, str(conf.round(2)))
                img = cv2.putText(img, text_for_vis, (int(xtext), int(ytext)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb_colors[label], 2,)
                if agree:
                    time_detect = datetime.datetime.now(pytz.timezone("America/New_York")).replace(tzinfo=None).strftime("%m-%d-%y %H:%M:%S")
                    cropped_image = img[ymin:ymax, xmin:xmax]
                    #cropped_image = resize(cropped_image, 200, 200)
                    retval, buffer_img= cv2.imencode('.jpg', cropped_image)
                    data = base64.b64encode(buffer_img).decode("utf-8")
                    html = "<img src='data:image/jpg;base64," + data + f"""' style='display:block;margin-left:auto;margin-right:auto;width:200px;border:0;'>"""
                    with lock:
                        result_queue.insert(0, {'object': class_, 'confident': str(conf.round(2)), 'img': html})
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
    col1, col2 = st.columns([2, 1])
    with col1:
        ctx = webrtc_streamer(key="example", video_frame_callback=transform,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": True, "audio": False}, mode=WebRtcMode.SENDRECV, async_processing=True)
        agree = st.checkbox("Enable clothes logging", value=False)
    with col2:
        labels_placeholder = st.empty()
    if agree:
        if ctx.state.playing:
            while True:
                time.sleep(0.5)
                with lock:
                    result_queue = result_queue[:5]
                    df = pd.DataFrame(result_queue)
                    st.markdown(hide_table_row_index , unsafe_allow_html=True)
                    labels_placeholder.write(df.to_html(escape=False), unsafe_allow_html=True)
    
def func_video(model):
    RTC_CONFIGURATION = RTCConfiguration(
       {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ) 
    #ctx1 = get_report_ctx()
    #print(ctx1.session_id)
    result_queue = []
    
    global frames_count
    frames_count = 0

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
        with lock:
            global frames_count 
            frames_count+=1
        
        #print(frames_count)
        if frames_count%4!=0:
            return frame
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
                img = cv2.rectangle(img, p0, p1, rgb_colors[label], 2) 
                ytext = ymin - 10 if ymin - 10 > 10 else ymin + 15
                xtext = xmin + 10
                class_ = CLASSES[label]
                text_for_vis = '{} {}'.format(class_, str(conf.round(2)))
                img = cv2.putText(img, text_for_vis, (int(xtext), int(ytext)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb_colors[label], 2,)
                if agree:
                    time_detect = datetime.datetime.now(pytz.timezone("America/New_York")).replace(tzinfo=None).strftime("%m-%d-%y %H:%M:%S")
                    cropped_image = img[ymin:ymax, xmin:xmax]
                    #cropped_image = resize(cropped_image, 200, 200)
                    retval, buffer_img= cv2.imencode('.jpg', cropped_image)
                    data = base64.b64encode(buffer_img).decode("utf-8")
                    html = "<img src='data:image/jpg;base64," + data + f"""' style='display:block;margin-left:auto;margin-right:auto;width:200px;border:0;'>"""
                    with lock:
                        result_queue.insert(0, {'object': class_, 'confident': str(conf.round(2)), 'img': html})
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
    path = ''
    uploaded_video = st.file_uploader("Upload video", type = ['mp4','mpeg','mov'], accept_multiple_files=False)
    if uploaded_video != None:
        vid = uploaded_video.name
        last_name = path.replace('data/','')
        if last_name!=vid:
            filelist = [f for f in os.listdir('data')]
            for f in filelist:
                os.remove('data/' + f)
        path = 'data/' + vid
        with open(path, mode='wb') as f:
            f.write(uploaded_video.read())
    col1, col2 = st.columns([2, 1])
    with col1:
        ctx_l = webrtc_streamer(
            key="key",
            mode=WebRtcMode.RECVONLY,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={
                "video": True,
                "audio": False,
            },
            player_factory=create_player,
            video_frame_callback=transform)
        agree = st.checkbox("Enable clothes logging", value=False)
    with col2:
        labels_placeholder = st.empty()
    if agree:
        if ctx_l.state.playing:
            while True:
                time.sleep(0.5)
                with lock:
                    result_queue = result_queue[:5]
                    df = pd.DataFrame(result_queue)
                    st.markdown(hide_table_row_index , unsafe_allow_html=True)
                    labels_placeholder.write(df.to_html(escape=False), unsafe_allow_html=True)

if __name__ == "__main__":
    main()