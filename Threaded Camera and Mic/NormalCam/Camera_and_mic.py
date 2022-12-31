########################################################All Imports###############################################################
from __future__ import division
import pyaudio
from google.cloud import speech
import os
import sys
import re
from six.moves import queue
import numpy as np
import socket
import threading
import cv2
import time
import math as m
import usb.core
import usb.util
import pyttsx3 as pyttsx
from tuning import Tuning
from FaceRecognition_NormalCam import Face_Recognition
from QRDetection import QR_Code_Detection
from Chat import ChatBot
###################################################################################################################################

########################################################parameteres################################################################
SPEAKER_RATE = 16000
SPEAKER_CHANNELS = 1 # change base on firmwares, 1_channel_firmware.bin as 1 or 6_channels_firmware.bin as 6
SPEAKER_WIDTH = 2
# run getDeviceInfo.py to get index
SPEAKER_INDEX = 0  # refer to input device id
CHUNK = 1024
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"
angle=None
QRangle=None
#########################################################Different Threads for running the intelligence###########################
video_thread=None
audioThread=None
QR_Thread=None
angleThread=None
stop_event=threading.Event()
#################################################################################################################################
HFOV=90
VFOV=65
##################################################################################################################################

########################################Initialize USB Device#####################################################
#dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
##################################################################################################################


###############Function for Communication between Jetson Nano and Jetson Xavier###############
'''
def Connection(ang):
    print("Server Starting")
    client=socket.socket(family=socket.AF_INET,type=socket.SOCK_STREAM)
    client.connect(ADDR)
    #file=open('data/Message.txt','r')
    data=str(ang)
    
    client.send('Message.txt'.encode('utf-8'))
    msg=client.recv(1024).decode('utf-8')
    print(f'Server: {msg}')
    
    client.send(data.encode('utf-8'))
    msg=client.recv(1024).decode('utf-8')
    print(f'Server: {msg}')
    
    #file.close()
    client.close()

'''
###############################################################################################

################Function for finding angle so that when keyword is said the robot turn to that angle#########################
def getAngle():
    if dev:
        Mic_tuning = Tuning(dev)
        while True:
            angle=Mic_tuning.direction


############################ OOP based method of starting camera in a thread to run camera and mic seperately####################
class Camera():
    def __init__(self):
        self.index=0
        self.arr=[]
        self.i=10
        while self.i > 0:
            cap=cv2.VideoCapture(self.index)
            if cap.read()[0]:
                self.arr.append(self.index)
                cap.release()
            self.index+=1
            self.i-=1
        print(self.arr)
        self.cap=cv2.VideoCapture(self.arr[0])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.FPS=self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Initizalied Camera at {self.FPS}")
    
    def getFrame(self):
        ret,img=self.cap.read()
        return img

    def record(self):
        while True:
            ret,frame=self.cap.read()
            if ret:
                cv2.imshow('Camera',frame)
                cv2.waitKey(1)
            # if cv2.waitKey(1) & 0xFF==ord('q'):
            #     break
    def start(self):
        global video_thread
        video_thread=threading.Thread(target=self.record)
        video_thread.start()

    def stop(self):
        global video_thread
        print("Closed Video thread")
        video_thread.join()
        cv2.destroyAllWindows()
        self.cap.release()
########################################################################################################################



#################################Microphone initialization using PyAudio for GCP STT#####################################
####################################Code is taken from their website###################################################
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
            channels=SPEAKER_CHANNELS,
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

########################################################################################################################

################################################QR Code Detection#######################################################
def calcAngle(x,y):
    hangle=((x-640)/(640))*(HFOV/2)
    vangle=((y-360)/(360))*(VFOV/2)
    return m.sqrt(hangle**2+vangle**2)


def DetectQR(stop_event):
    global QRangle
    global QR_Thread
    while not stop_event.is_set():
        frame1=camera.getFrame()
        try:
            qr.preprocess(frame1)
            processedImg=qr.postProcess(frame1)
            centre=qr.getCentre()
            if centre is not None:
                QRangle=calcAngle(centre[0],centre[1])
                cv2.circle(frame1,centre,4,(0,255,255),2)
                print(QRangle)
        except:
            pass

#########################################Function which sends gRPC requests and performs STT############################
def listen_print_loop(responses):
    global QR_Thread
    global stop_event
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
            #if re.search(r"\b(exit|quit)\b", transcript, re.I):
            #    print("Exiting..")
            #    break
            x=cb.Chat(transcript+overwrite_chars,camera.getFrame(),fr)
            if x=='QR':
                time.sleep(3)
                QR_Thread.start()
            if x=='Stop QR':
                if QR_Thread.is_alive():
                    stop_event.set()
                    print("Closed QR Thread")                
            if x=='Location':
                print('Location')
            if x=='exit':
                break
            num_chars_printed = 0
###########################################################################################################################

###################################################Code for initialzing Microphone and calling STT function################
def audioMain():
    # See http://g.co/cloud/speech/docs/languages
    # for a list of supported languages.
    language_code = "en-IN"  # a BCP-47 language tag
    client = speech.SpeechClient.from_service_account_file('/home/aakash/Desktop/GCP_STT_Cred.json')
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SPEAKER_RATE,
        audio_channel_count=SPEAKER_CHANNELS,
        language_code='en-IN'
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )
    with MicrophoneStream(SPEAKER_RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (
            speech.StreamingRecognizeRequest(audio_content=content)
            for content in audio_generator
        )
        responses = client.streaming_recognize(streaming_config, requests)
        # Now, put the transcription responses to use.
        listen_print_loop(responses)
##############################################################################################################################
if __name__=='__main__':
    cb=ChatBot()
    camera=Camera()
    camera.start()
    fr=Face_Recognition()
    qr=QR_Code_Detection()
    #audioThread=threading.Thread(target=audioMain)
    #angleThread=threading.Thread(target=getAngle)
    QR_Thread=threading.Thread(target=DetectQR,args=(stop_event,))
    audioMain()
    camera.stop()
    if QR_Thread.is_alive():
        QR_Thread.exit()
        print("Closed QR Thread")