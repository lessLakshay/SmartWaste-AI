import os
os.system("pip install -r requirements.txt")

import streamlit as st
import cv2
import threading
import time
from ultralytics import YOLO
import cvzone
import math
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import gdown


#downloading yolov8 paramenters saved on drive 
MODEL_URL = "https://drive.google.com/uc?id=1ZYSRdRpFwjyYEpMMDtrpBqXgiQQe_oam"
MODEL_PATH = "best.pt"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model weights..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

sender_email = None
reciever_email = None
app_pass = None

msges = []

#function to send automated email
def send_alert_email(area_name, frame):
    """
    Sends an email alert with an attached image frame showing garbage detection.
    """

    # ---- CONFIG ----
    # Create from Google > Manage Your Account > App Passwords
    subject = f"Garbage Detected in {area_name}"
    
    body = f"""
    Dear Maintenance Team,

    The monitoring system has detected garbage in **{area_name}**.

    Attached is an image frame captured from the camera feed showing the detected garbage.
    
    Please send the cleaning staff to inspect and clean the area immediately.

    Regards,  
    Smart Garbage Detection System  
    """

    # convert frame (opencv image) to jpeg bytes 
    _, img_encoded = cv2.imencode('.jpg', frame)
    image_bytes = img_encoded.tobytes()

    # compose email 
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = reciever_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    # attach Image
    image_part = MIMEImage(image_bytes, name=f"garbage_{area_name}.jpg")
    msg.attach(image_part)

    # send 
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, app_pass)
        server.sendmail(sender_email, reciever_email, msg.as_string())
        server.quit()
        print(f"Email with image sent successfully to {reciever_email} for area {area_name}")
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False


model = YOLO("best.pt")
classes = ['0', 'c', 'garbage', 'garbage_bag', 'sampah-detection', 'trash']

#function to detect garbaeg in video frames
def detect_garbage(frame):
    garbage_detected = False
    output = model(frame, stream=True)
    for r in output:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            if conf > 0.1:
                #drawing boundary boxes
                cvzone.cornerRect(frame, (x1, y1, w, h), t=2)
                cvzone.putTextRect(frame, f'{classes[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
                garbage_detected=True
    return frame, garbage_detected

# Video Stream Thread Class
class VideoStream:
    def __init__(self, source, area_name):
        self.source = source
        self.area_name = area_name
        self.cap = cv2.VideoCapture(source)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        self.last_detect_time = 0
        self.garbage_detected = False
        self.alert_sent = False
        self.last_garbage_time = 0
        self.msg = None

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(1)
                continue

            current_time = time.time()
            # Perform garbage detection every 3 seconds
            if current_time - self.last_detect_time >= 3:
                frame, garbage_detected = detect_garbage(frame)
                self.garbage_detected = garbage_detected
                self.last_detect_time = current_time

                if garbage_detected:
                    if self.last_garbage_time == 0:
                        self.last_garbage_time = current_time
                    elif (current_time - self.last_garbage_time > 5) and not self.alert_sent:
                        send_alert_email(self.area_name, frame)
                        self.alert_sent = True
                        self.msg = f"Garbage detected in {self.area_name} at {time.strftime('%H:%M:%S')}. Email Has Been Sent!"
                        print('f')
                else:
                    last_garbage_time = 0
                    self.alert_sent=False

            with self.lock:
                self.frame = frame

            time.sleep(0.03)

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        self.cap.release()

    def getMsg(self):
        return self.msg
    
    def resetMsg(self):
        self.msg = None

# Streamlit UI
st.set_page_config(page_title="Garbage Detection Dashboard", layout="wide")

if "streams" not in st.session_state:
    st.session_state.streams = {}
if "running" not in st.session_state:
    st.session_state.running = False

st.sidebar.title("Add Camera Stream")
url = st.sidebar.text_input("Enter video / RTSP URL (e.g., 0 for webcam)")
area_name = st.sidebar.text_input("Enter area name")
add_btn = st.sidebar.button("Add Stream")

st.sidebar.title("Add Credentials")
sender_email = st.sidebar.text_input("Enter Sender Email (Your Email)")
app_pass = st.sidebar.text_input("Enter App Password")
reciever_email = st.sidebar.text_input("Enter Reciever Email")
st.sidebar.write("Email Wont Be Sent If Credentials Arent Valid!! ")


if add_btn:
    if url and area_name:
        st.session_state.streams[area_name] = VideoStream(url, area_name)
        st.sidebar.success(f"Added stream for {area_name}")
    else:
        st.sidebar.error("Please provide both URL and area name!")

st.title("Real-Time Camera Dashboard")

# Display active streams
if not st.session_state.streams:
    st.info("No active streams. Add one from the sidebar.")
else:
    cols = st.columns(2)
    placeholders = {}
    for i, (area, stream) in enumerate(st.session_state.streams.items()):
        with cols[i % 2]:
            st.subheader(f"{area}")
            placeholders[area] = st.empty()

    st.session_state.running = True

    # Live Frame Update Loop 

    while st.session_state.running:
        for area, stream in st.session_state.streams.items():
            frame = stream.get_frame()
            if stream.getMsg()!=None:
                st.success(stream.getMsg())
                stream.resetMsg()
            
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                placeholders[area].image(frame, channels="RGB", use_container_width=True)
        time.sleep(0.05)  # Refresh ~20 FPS



