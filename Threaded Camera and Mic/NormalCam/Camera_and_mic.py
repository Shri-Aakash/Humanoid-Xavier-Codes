from __future__ import division
import pyaudio
from google.cloud import speech
import os
import sys
import re
from six.moves import queue
import numpy as np
import threading
import cv2
import time
from tuning import Tuning
from FaceRecognition_NormalCam import Face_Recognition
from QRDetection import QR_Code_Detection
from Chat import ChatBot


SPEAKER_RATE = 16000
SPEAKER_CHANNELS = 1 # change base on firmwares, 1_channel_firmware.bin as 1 or 6_channels_firmware.bin as 6
RESPEAKER_WIDTH = 2
# run getDeviceInfo.py to get index
RESPEAKER_INDEX = 0  # refer to input device id
CHUNK = 1024
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"
angle=None
video_thread=None

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
		self.cap=cv2.VideoCapture(self.arr[-1])
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
			if cv2.waitKey(1) & 0xFF==ord('q'):
				break

	def start(self):
		global video_thread
		video_thread=threading.Thread(target=self.record)
		video_thread.start()

	def stop(self):
		global video_thread
		video_thread.join()
		self.cap.release()

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
            channels=RESPEAKER_CHANNELS,
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
            	
            if x=='exit':
                break
            num_chars_printed = 0


os.environ['GOOGLE_CREDENTIALS']='/home/aakash/Desktop/GCP_STT_Cred.json'

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


if __name__=='__main__':
	cb=ChatBot()
    camera=Camera()
    camera.start()
    fr=Face_Recognition()
    qr=QR_Code_Detection()
    main()
