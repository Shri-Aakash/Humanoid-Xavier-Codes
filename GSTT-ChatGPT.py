# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 17:35:21 2022

@author: aakaa
"""
from __future__ import division
import pyaudio
from google.cloud import speech
import os
import sys
import re
from six.moves import queue
from pyChatGPT import ChatGPT
import pyttsx3 as pyttsx
engine=pyttsx.init()
voices=engine.getProperty('voices')
engine.setProperty('voice',voices[0].id)
engine.setProperty('rate',150)
engine.runAndWait()
session_token='eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIn0..viyKDk5K2xfVNiR2.6Y7SfqEDHFxD7GfsNF1e0QbVztXvlRwqP-TeRT1Dw7Q8QdWCgtHKvkKMC_3LOMnGpeaVlJ3xQ2rDQEysDPVoB6BZ1b3M6FoJxWpJvFFsAhoCXlrhS7rKpBcBV8Rv81i4LOb7Quh62nrMPN10SlnTB7DeALHHFe6b2Wq5Oh-p4TWjIiPpJUXgCr76hYfNxOc4b_l1gxGI9VkQQeExJBzW4KTYej0XhTpDWnfLQSzs3C2Rkh2iHveFX_s9FrQgo-HnB8sCC6vNaOL9VaqPRJBgj5mknPyle1p4e76XC-GMz_PRtudVCfSc_z1xw2IHHaKl2yYva5i6xCCU-7Pibt2za4340HL0P4kYsC3yiogkY_A6183572DU79VcRmGkzn-q0SMaTg0bXwtKaCbaoK-cpfyO0rkIC_CKuULUTX6TT22hiUFVLem5PX1ztvG8MGyD3Y03smGYYJ-aJ2kR6ubSu8gUoBvdosNaQp7EfvEC1yNl3ybPCFG7eoZbw_ttgu0n-x6ezgYZrOjnfzWtxTO4FMTB3NSrz5xIwyMxWMsdyHTSKki04q1c2YSzzsHao7vywhOWJ5-cugDyAOoHPJYa3K9w_1BHdHZfPu93G1PbxlwCPr3IkXG3pmO_gjgo6znfDrLnOeHCHF0HC7ubSQx0P3jUZmwwhsC5DwdzEv6U0fkRQAhICfseNej_3gbj-dXE8N_d0h6TSH0Fq70JCVzCE8KMMS5yDkLcey89bv9uLc6YFsZkI7q_jTUipai8V3u8QKPvXjIeojrUkLw4b5p37H9751qmLYEqsEhr-iJnXszDEOpJ0f2NYz_VSY0OPrOLQL9MeW-WksHK4L601OWWQGqiP-63PXJnMXUerDPcjSuZOweZn_Q5EWOOl3-0XVkP1il8H5UFQ1DXu2ULPC2t80MB_LL7HhiHsR8NxRYaUy0p4DxCVmEcAsRRFOz0wdWwLC52MU15SqVUpLn_pBfl0N3pvKUNcH_nB3JrXKtZuYitlhiEGxp9P0f6jkFyYDjdwA-EIt14NwIKYqU3v_w3dY_Hw9_4ySsa4wolQacXtRPNii0KiR32guvSUhcldPhuy3llFTBwaAqJEIhfH31SxIHT4HxSNLx6MLAF2JPgC6JKOEYCscT4IZLBQxx9MJQEtgE1DQyRlEX7sARmPdxlv1ZsDLFvYIOQHxu6rsPlffyGz0-FyAeM-kKbD5AD0hgvqI-P3hey8OI22LqS0kqeLDq80z9T-mL7Bi-XlukoPKU83J8uGGl_YiqCst8xrQ6xMSAIOo7IeRTubrEkvHNkLXbNLTViGuaaYamdn6ystEaZjmMy15hlPRGr7ybAiHr2qki_MFJ22PXPaDnpyxQdMkyhHTByc6lncI_7s2lP5JRD2UyiaNUKsMhN3os7LepyOgN6FZ2kOHu4YNJlsZNzuLviy-H5_BCJHUbaMqkhO_FGs68-dif4m4B9chTw4rKUTQdg07xfVwaYLbrZews2KK4NbSdkg4l6cpz4PDQC8RFgfo985Pw153ijImobsxTa3APFQzKnFczdliYL9OLfy9VyvLdfUSq1eUC-JeDBTUQ4zKcJQGKi9PPe6xWHC5sWf7nBd-rBzJ2DjiPxIPZO9jZ1BKlVmbNRepGN22CgiLMShWRrK1EhAlUB5oA6gnFM9nGbgUft_2_LvZfXFVjB1ZouJ4s6VonujHcQBz28HRkudA_DMQ4Ba8EfEb-tHK0v2jSirSSQVHJrgHkBo8fIwtqmk0CE9ms0SdQpatZanFrcRf0puAzmcKZ0OssWpIFBhCgUkMvNky6FzseSI8nvek_TUMQIlTpEkX4_Rs2PzCDdNVD-y4NVuYrEMqiwOFWkZtJhA8Orb7tpVUQYaXD3f1qVWQWh2_E0RaoqIyC9MHyQ3ulJtD9J3t-v8dAiwBJ1ptcCl63BrmR-bB4PHaSlELCVZYvhC92Gop-xXBDom2l15PoTQUA8Jbp0bhbXuxTv6xAuZecMByzBhg6HxerAJO_s2BeH4gFi1J6fOzyM-nEpufI5A_qcvMyqAEc1A9yTF_Rx4MpEwh10OXeYHwKDoL9y5xB-VU2nmTEYrC1CxPBQjBm8z9PRK0t8ejdJXanKMOm_8x-RyWtRZsAGMeY6JKj0z3iZdPfyc6Fr1JV8oWs4ofyoCNo3AkYN5ylm3LA0ObYxQNsy8Yh-xaGJKLTGPqs5gR9XOodAs1vd6s6iHBVBXMCB9UikL2uc48VYzEX7_i5CKmdnEku235C2g9omu4kEOgvtszkBu6NT5d54o3Y2Nok61tSMpgINsJyuLcwPfRABqyXfLfIBVokBrCSGnMG_ky4oImmPhtcK3pWV_G-Lc7o9PaHcJhqxhC4bKvBbZn_NL9TGF9frxQ_nC6Ufu09UGWc.71Ox0ERSNLdqmpOlgvQSJg'
api=ChatGPT(session_token) 
RESPEAKER_RATE = 16000
RESPEAKER_CHANNELS = 1 # change base on firmwares, 1_channel_firmware.bin as 1 or 6_channels_firmware.bin as 6
RESPEAKER_WIDTH = 2
# run getDeviceInfo.py to get index
RESPEAKER_INDEX = 7  # refer to input device id
CHUNK = 1024
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"

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
            channels=1,
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
            if re.search(r"\b(exit|quit)\b", transcript, re.I):
                print("Exiting..")
                break
            resp=api.send_message(transcript + overwrite_chars)
            print('User: ',transcript + overwrite_chars)
            print('ChatGPT: ',resp['message'])
            engine.say(resp['message'])
            engine.runAndWait()
			# Exit recognition if any of the transcribed phrases could be
			# one of our keywords.
            num_chars_printed = 0
os.environ['GOOGLE_CREDENTIALS']='GCP Credentials.json'
client=speech.SpeechClient.from_service_account_file('GCP_STT_Cred.json')
config=speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=RESPEAKER_RATE,
    audio_channel_count=1,
    language_code='en-IN'
)

streaming_config = speech.StreamingRecognitionConfig(
        config=config,interim_results=True
)

with MicrophoneStream(RESPEAKER_RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (
            speech.StreamingRecognizeRequest(audio_content=content)
            for content in audio_generator
        )
        response = client.streaming_recognize(streaming_config, requests)
            # Now, put the transcription responses to use.
        listen_print_loop(response)

