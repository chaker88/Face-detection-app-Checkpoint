from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
import numpy as np
import streamlit as st
import threading
from typing import Union

cascade = cv2.CascadeClassifier("face_detection/haarcascade_frontalface_default.xml")
st.title("Face Detection")

class VideoTransformer():
    frame_lock: threading.Lock # transform() is running in another thread, then a lock object is used here for thread-safety.
    in_image: Union[np.ndarray, None]
    def __init__(self) -> None:
        self.frame_lock = threading.Lock()
        self.in_image = None
        self.scale= None
        self.minNeighbors=None
        self.RectColor=None
        
    
    def recv(self, frame)-> av.VideoFrame:
        frm = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, self.scale, self.minNeighbors) # type: ignore
        
        for x,y,w,h in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (224, 224))
            face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
            face = np.reshape(face, [1, 224, 224, 3])/255.0	
            cv2.rectangle(frm, (x,y), (x+w, y+h),self.RectColor , 2)  # type: ignore
            cv2.putText(frm, 'Face test', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.RectColor, 2) # type: ignore
        
        with self.frame_lock:
            self.in_image=frm

            
        return av.VideoFrame.from_ndarray(frm, format='bgr24')

ctx=webrtc_streamer(key="key", video_processor_factory=VideoTransformer, # type: ignore
				rtc_configuration=RTCConfiguration(
					{"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
					)
	)

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    bgr_color = tuple(int(hex_color[i:i + 2], 16) for i in (4, 2, 0))  # Reversing the order to convert from RGB to BGR
    return bgr_color

if ctx.video_transformer:
    if st.button("snapshot"):
        with ctx.video_transformer.frame_lock: # type: ignore
            frame = ctx.video_transformer.in_image # type: ignore
            
        
        if frame is not None: 
            st.write("Input image:")
            st.image(frame, channels="BGR")
            cv2.imwrite('face_detection/frame_on_button_click.png',frame)
        else:
            st.warning("No frames available yet.")
    color = st.color_picker('Pick A Color', '#df4cc5')
    rgb = hex_to_bgr(color)
    ctx.video_transformer.RectColor = rgb
    st.write(rgb)
    values_slide = st.slider('Select the scale value',1.0, 1.8, 1.1)
    ctx.video_transformer.scale = values_slide
    values_min_neighboor = st.slider('Select the minNeighbors value',1, 10, 5)
    ctx.video_transformer.minNeighbors = values_min_neighboor

    
#if ctx.video_processor:
