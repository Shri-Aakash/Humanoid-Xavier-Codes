from __future__ import division
import pyaudio
from google.cloud import speech
import os
import sys
import re
from six.moves import queue
import pyrealsense2 as rs
from gtts import gTTS
import usb.core
import usb.util
import numpy as np
import threading
import math as m
import usb.core
import usb.util
import cv2
import time
import socket
from FaceRecognition import Face_Recognition
from QRDetection import QR_Code_Detection
from Chat import ChatBot
from tuning import Tuning
#############################################Initialized ReSpeaker Mic#########################################################
dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
##############################################################################################################################

##############################################ReSpeaker Parameters############################################################
RESPEAKER_RATE = 16000
RESPEAKER_CHANNELS = 6 # change base on firmwares, 1_channel_firmware.bin as 1 or 6_channels_firmware.bin as 6
RESPEAKER_WIDTH = 2
# run getDeviceInfo.py to get index
RESPEAKER_INDEX = 0  # refer to input device id
CHUNK = 1024
RECORD_SECONDS = 5
Output_FN = "output.mp3"
###############################################################################################################################

####################################################### Different values for caliberation #######################################
angle=None
QRangle=None
distance=None
################################################################################################################################

####################################################### Camera FOV #############################################################
HFOV=90
VFOV=65
################################################################################################################################

####################################################### Different threads for running different processes#####################
video_thread=None
audioThread=None
angleThread=None
stop_event=False
##############################################################################################################################

###############Function for Communication between Jetson Nano and Jetson Xavier###############
def Connection(l):
    print("Server Starting")
    client=socket.socket(family=socket.AF_INET,type=socket.SOCK_STREAM)
    client.connect(ADDR)
    #file=open('data/Message.txt','r')
    if len(l)==2:
        if l[0]=='Mic':
            data=str(l[1])
            client.send('Message_Mic.txt'.encode('utf-8'))
            msg=client.recv(1024).decode('utf-8')
            print(f'Server: {msg}')
            client.send(data.encode('utf-8'))
            msg=client.recv(1024).decode('utf-8')
            print(f'Server: {msg}')

    if len(l)==3:
        if l[0]=='QR':
        data=' '.join(l[1:])
        client.send('Message_Cam.txt'.encode('utf-8'))
        msg=client.recv(1024).decode('utf-8')
        print(f'Server: {msg}')
        client.send(data.encode('utf-8'))
        msg=client.recv(1024).decode('utf-8')
        print(f'Server: {msg}')
    
    #file.close()
    client.close()
###############################################################################################

######################################Set of Functions for detecting QR and calculating angle and distance to centre####################
def calcAngle(x,y):
    hangle=((x-640)/(640))*(HFOV/2)
    vangle=((y-360)/(360))*(VFOV/2)
    return m.sqrt(hangle**2+vangle**2)


def DetectQR():
    global QRangle
    while True:
        ret,color_frame,depth_frame=dc.get_frame()
        try:
            qr.preprocess(color_frame)
            processedImg=qr.postProcess(color_frame)
            centre=qr.getCentre()
            if centre is not None:
                QRangle=calcAngle(centre[0],centre[1])
                cv2.circle(color_frame,centre,4,(0,255,255),2)
                print(QRangle)
                distance=depth_frame[int(centre[1]),int(centre[0])]
                if distance<0.1:
                    Connection(['QR',0,0])
                    break
                else:
                    if round(QRangle)!=0:
                        Connection(['QR',1,0])
                    else:
                        Connection(['QR',0,1])
        except:
            pass
#######################################################################################################################################

################################################Class For IntelRealsense Depth Camera###########################################
class DepthCamera:
    def __init__(self):
        # Configure depth and color streams
        self.width=1280
        self.height=720
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(config)
        self.started=True


    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        if not depth_frame or not color_frame:
            return False, None, None
        return True, depth_image, color_image

    def Get_color_frame(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        if not depth_frame or not color_frame:
            return False, None, None
        return color_image

    def show(self):
        while True:
            ret,depthFrame,colorFrame=self.get_frame()
            cv2.imshow('Color Frame',colorFrame)
            if cv2.waitKey(1) & 0xFF==ord('q'):
                break

    def start(self):
        global video_thread
        video_thread = threading.Thread(target=self.show)
        video_thread.start()

    def read(self):
        color_image=np.asanyarray(self.color_frame.get_data())
        depth_image=np.asanyarray(self.depth_frame.get_data())
        return True,color_image,depth_image

    def release(self):
        global video_thread
        self.pipeline.stop()
        self.started=False
        video_thread.join()
###########################################################################################################################


###########################################Class For initializing Microphone using pyaudio####################################
class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=6,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)
###############################################################################################################################


###################################################Function for calling GCP STT API using gRPC#################################
def listen_print_loop(responses):
    """Iterates through server responses and prints them.

    The responses passed is a generator that will block until a response
    is provided by the server.

    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.

    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.
    """
    global angle
    num_chars_printed = 0
    for response in responses:
        if not response.results:
            continue

        # The `results` list is consecutive. For streaming, we only care about
        # the first result being considered, since once it's `is_final`, it
        # moves on to considering the next utterance.
        result = response.results[0]
        if not result.alternatives:
            continue

        # Display the transcription of the top alternative.
        transcript = result.alternatives[0].transcript

        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        #
        # If the previous result was longer than this one, we need to print
        # some extra spaces to overwrite the previous result
        overwrite_chars = " " * (num_chars_printed - len(transcript))

        if not result.is_final:
            sys.stdout.write(transcript + overwrite_chars + "\r")
            sys.stdout.flush()

            num_chars_printed = len(transcript)

        else:
            print(transcript + overwrite_chars)

            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            if re.search(r"\b(Hi Alpha|Hey Alpha|Alpha)\b",transcript,re.I):
                print(angle)
                Connection(['Mic',angle])

            x=cb.Chat(transcript+overwrite_chars,dc.Get_color_frame(),fr)
            if x[0]=='':
                if x[1]=='exit':
                    break
                if x[1]=='stopQR':
                    QR_Thread.join()
            else:
                if x[1]=='':
                    tts=gTTS(x[0],lang='en')
                    tts.save(Output_FN)
                    os.system("mpg123 "+Output_FN)
                    time.sleep(5)
                if x[1]=='QR':
                    tts=gTTS(x[0],lang='en')
                    tts.save(Output_FN)
                    os.system("mpg123 "+Output_FN)
                    time.sleep(5) 
                    DetectQR()
                if x[1]=='Location':
                    tts=gTTS(x[0],lang='en')
                    tts.save(Output_FN)
                    os.system("mpg123 "+Output_FN)
                    time.sleep(5)              
                if x[1]=='-':
                    tts=gTTS(x[0],lang='en')
                    tts.save(Output_FN)
                    os.system("mpg123 "+Output_FN)
                    time.sleep(5)
                if x[1]=='Face not visible':
                    tts=gTTS(x[0],lang='en')
                    tts.save(Output_FN)
                    os.system("mpg123 "+Output_FN)
                    time.sleep(5)
                if x[1]=='NaN':
                    tts=gTTS(x[0],lang='en')
                    tts.save(Output_FN)
                    os.system("mpg123 "+Output_FN)
                    time.sleep(5)
            num_chars_printed = 0
##############################################################################################################################


############################################Function for Configuring STT parameters and taking audio through Mic#############
def audioMain():
    language_code = "en-IN"  # a BCP-47 language tag

    client = speech.SpeechClient.from_service_account_file('/home/irp2022/Desktop/GCP_STT_Cred.json')
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RESPEAKER_RATE,
        audio_channel_count=6,
        language_code='en-IN'
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    with MicrophoneStream(RESPEAKER_RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (
            speech.StreamingRecognizeRequest(audio_content=content)
            for content in audio_generator
        )

        responses = client.streaming_recognize(streaming_config, requests)

        # Now, put the transcription responses to use.
        listen_print_loop(responses)
###########################################################################################################################

if __name__=='__main__':
    dc=DepthCamera()
    dc.start()
    angle_thread=threading.Thread(target=getAngle)
    angle_thread.start()
    audio_thread=threading.Thread(target=audioMain)
    dc.release()
    angle_thread.join()