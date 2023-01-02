import cv2
import numpy as np
#from FaceRecognition_NormalCam import Camera
# from realsense_depth import *


class QR_Code_Detection():
	INPUT_WIDTH=640
	INPUT_HEIGHT=640

	SCORE_THRESHOLD=0.5
	NMS_THRESHOLD=0.45
	CONFIDENCE_THRESHOLD=0.45

	FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
	FONT_SCALE = 0.7
	THICKNESS = 1

	# Colors.
	BLACK = (0, 0, 0)
	BLUE = (255, 178, 50)
	YELLOW = (0, 255, 255)	

	modelWeights = "best.onnx"
	net = cv2.dnn.readNet(modelWeights)

	def __init__(self):
		QR_Code_Detection.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
		QR_Code_Detection.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
		self.classes=['QR']
		self.centre=None

	def drawlabel(self,im, label, x, y):
		text_size = cv2.getTextSize(label, QR_Code_Detection.FONT_FACE, QR_Code_Detection.FONT_SCALE, QR_Code_Detection.THICKNESS)
		dim, baseline = text_size[0],text_size[1]
    	# Use text size to create a BLACK rectangle.
		cv2.rectangle(im,(x,y),(x+dim[0],y+dim[1]+baseline),(0,0,0),cv2.FILLED)
    	# Display text inside the rectangle.
		cv2.putText(im, label, (x, y + dim[1]), QR_Code_Detection.FONT_FACE, QR_Code_Detection.FONT_SCALE, QR_Code_Detection.YELLOW, QR_Code_Detection.THICKNESS, cv2.LINE_AA)

	def preprocess(self,input_image):
		self.blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (QR_Code_Detection.INPUT_WIDTH, QR_Code_Detection.INPUT_HEIGHT), [0, 0, 0], 1, crop=False)
    	# Sets the input to the network.
		QR_Code_Detection.net.setInput(self.blob)

    	# Run the forward pass to get output of the output layers.
		self.outputs = QR_Code_Detection.net.forward(QR_Code_Detection.net.getUnconnectedOutLayersNames())

	def postProcess(self,input_image):
		self.class_ids = []
		self.confidences = []
		self.boxes = []
    	# Rows.
		self.rows = self.outputs[0].shape[1]
		self.image_height, self.image_width = input_image.shape[:2]
		# Resizing factor.
		self.x_factor = self.image_width / QR_Code_Detection.INPUT_WIDTH
		self.y_factor = self.image_height / QR_Code_Detection.INPUT_HEIGHT
		# Iterate through detections.
		for r in range(self.rows):
			self.row = self.outputs[0][0][r]
			self.confidence = self.row[4]
			# Discard bad detections and continue.
			if self.confidence >= QR_Code_Detection.CONFIDENCE_THRESHOLD:
				self.classes_scores = self.row[5:]
				# Get the index of max class score.
				self.class_id = np.argmax(self.classes_scores)
				#  Continue if the class score is above threshold.
				if (self.classes_scores[self.class_id] > QR_Code_Detection.SCORE_THRESHOLD):
					self.confidences.append(self.confidence)
					self.class_ids.append(self.class_id)
					self.cx, self.cy, self.w, self.h = self.row[0], self.row[1], self.row[2], self.row[3]
					self.left = int((self.cx - self.w / 2) * self.x_factor)
					self.top = int((self.cy - self.h / 2) * self.y_factor)
					self.width = int(self.w * self.x_factor)
					self.height = int(self.h * self.y_factor)
					self.box = np.array([self.left, self.top, self.width, self.height])
					self.boxes.append(self.box)
		self.indices = cv2.dnn.NMSBoxes(self.boxes, np.array(self.confidences), QR_Code_Detection.CONFIDENCE_THRESHOLD, QR_Code_Detection.NMS_THRESHOLD)
		for i in self.indices:
			self.box = self.boxes[i]
			self.left = self.box[0]
			self.top = self.box[1]
			self.width = self.box[2]
			self.height = self.box[3]             
			# Draw bounding box.   
			self.centre=(self.left+self.width//2,self.top+self.height//2)          
			cv2.rectangle(input_image, (self.left, self.top), (self.left + self.width, self.top + self.height), QR_Code_Detection.BLUE, 3*QR_Code_Detection.THICKNESS)
			cv2.circle(input_image,self.centre,4,(0,255,255),2)
			# Class label.                      
			self.label = "{}:{:.2f}".format(self.classes[self.class_ids[i]],self.confidences[i])        
			# Draw label.             
			self.drawlabel(input_image, self.label, self.left, self.top)
		return input_image

	def getCentre(self):
		return self.centre



if __name__=='__main__':
	qr=QR_Code_Detection()
	camera=Camera()
	while True:
		frame1=camera.getFrame()
		qr.preprocess(frame1)
		processedImg=qr.postProcess(frame1)
		cv2.imshow('QR Code Detection',processedImg)
		centre=qr.getCentre()
		print(centre)
		if cv2.waitKey(1) & 0xFF==ord('q'):
			break
	camera.release()

