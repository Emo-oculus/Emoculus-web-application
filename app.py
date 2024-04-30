import numpy as np
import cv2
import streamlit as st
from tensorflow import keras
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode

# load model
emotion_dict = {0:'angry', 1 :'happy', 2: 'neutral', 3:'sad', 4: 'surprise'}
# load json and create model
json_file = open('emotion_model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)

# load weights into new model
classifier.load_weights("emotion_model1.h5")

#load face
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class Faceemotion(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        #image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)
            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img


def main():
    # Face Analysis Application #

    st.markdown("<h1 style='text-align: center; color: blue;font-size: 80px'>Em⊙-⊙culus</h1>", unsafe_allow_html=True)
    st.sidebar.markdown("<h1 style='text-align: center; color: blue;font-size: 40px'>Em⊙-⊙culus</h1>", unsafe_allow_html=True)
    activiteis = ["Home", "EMO-OCULUS", "About"]
    choice = st.sidebar.selectbox("Select Activity", activiteis)
    st.sidebar.markdown("Emo-oculus is a python based web application that can detect the user's emotions based on their facial expressions that are fed in once the user grants the access. emo-oculus classifies the user's expression into one of the seven basic expressions they are sad, happy, fear, anger, suprise, neutral and disgust."
       )
    st.sidebar.markdown("<h1 style='text-align: center; color: blue;font-size: 40px'>(⊙_⊙)</h1>", unsafe_allow_html=True)
    st.sidebar.markdown("<h1 style='text-align: center; color: blue;font-size: 40px'>.</h1>", unsafe_allow_html=True)
    st.sidebar.markdown("<h1 style='text-align: center; color: blue;font-size: 40px'>.</h1>", unsafe_allow_html=True)
    if choice == "Home":

        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px;border-radius: 10px;width: 500px; margin: 0 auto;">
                                            <h4 style="color:white;text-align:center;">
                                            REALTIME FACIAL EMOTION RECOGNITION</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)

        image_url_left = "https://www.pinpng.com/pngs/m/163-1630042_lpu-official-logo-hd-png-download.png"
        image_url_right = "https://assets.zyrosite.com/cdn-cgi/image/format=auto,w=379,h=336,fit=crop/AQEXp3M1wrIjxLox/upgrad_image-removebg-preview-mjELgyJGZvu1l2oq.png"

        # Display the images with circular crop aligned to left and right using HTML
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between;">
                <div style="margin-right: auto; margin-top: 20px;">
                    <img src="{image_url_left}" alt="Left Image" style="width: 300px; height: 300px; object-fit: cover; border-radius: 50%;">
                </div>
                <div style="margin-top: 20px; text-align: center;">
                    <h2>Developed by</h2>
                    <h2>AMAN::YASH::SREE</h2>
                    <h2>Under the guidance</h2>
                    <h2>Mr. Ajay Sharma</h2>
                </div>
                <div style="margin-left: auto; margin-top: 20px;">
                    <img src="{image_url_right}" alt="Right Image" style="width: 300px; height: 300px; object-fit: cover;">
                </div>
            </div>
            """,
           unsafe_allow_html=True
        )
    elif choice == "EMO-OCULUS":
        st.header("READY TO READ YOU :) ")
        st.write("grant the required permissions and make sure you have a good connection")
        webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                        video_processor_factory=Faceemotion)

    elif choice == "About":
        st.subheader("About this app")
        html_temp_about1 = """<div style="background-color:#6D7B8D;padding:10px">
                                    <h4 style="color:white;text-align:center;">
                                    Real time faceial emotion recognition application made using OpenCV, Tensorflow and Keras. Deployed with the help of streamlit</h4>
                                    </div>
                                    </br>"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)
        st.subheader("Contact us")
        html_temp4 = """
        <div style="background-color:#98AFC7;padding:10px; border-radius: 10px">
          <h4 style="color:white;text-align:center;">Github@Emo-oculus</h4>
          <h4 style="color:white;text-align:center;">emo.oculus.team@gmail.com</h4>
          <h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
        </div>
        <br></br>
        <br></br>
        """

        st.markdown(html_temp4, unsafe_allow_html=True)

    else:
        pass




if __name__ == "__main__":
    main()
