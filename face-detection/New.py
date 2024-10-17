from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
from pygame import mixer
import numpy as np

mixer.init()

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

def mouth_aspect_ratio(mouth):
	top_lip=mouth[50:55]
	top_lip=np.concatenate((top_lip,shape[61:65]))

	bottom_lip=mouth[56:60]
	bottom_lip=np.concatenate((bottom_lip,shape[65:68]))

	return sum(distance.euclidean(p1, p2) for p1 in top_lip for p2 in bottom_lip) / len(top_lip)

def midpoint(p1, p2):
  
  x1, y1 = p1
  x2, y2 = p2

  midpoint_x = (x1 + x2) / 2
  midpoint_y = (y1 + y2) / 2

  return [midpoint_x, midpoint_y]

def eye_distance(left_eye, right_eye):

  left_eye_center = np.mean(left_eye, axis=0)
  right_eye_center = np.mean(right_eye, axis=0)

  return np.abs(left_eye_center[0] - right_eye_center[0])

thresh_eye=0.25
thresh_mouth=40
frame_check_eye=20
frame_check_mouth=10
frame_check_direction=45


detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

cap=cv2.VideoCapture(0)

flag_eye=0
flag_mouth=0
flag_direction=0

while True:
	ret, frame=cap.read()
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.equalizeHist(gray) 
	gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=40)  
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	gray = clahe.apply(gray)
	subjects = detect(gray, 0)
	for subject in subjects:
		shape = predict(gray, subject)
		shape = face_utils.shape_to_np(shape)

		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		mouth= shape[mStart:mEnd]

		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		mouth_dist=mouth_aspect_ratio(mouth)
		ear = (leftEAR + rightEAR) / 2.0
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		mouthHull=cv2.convexHull(mouth)

		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame,[mouthHull],-1,(255,0,0),1)

		gaze_direction=eye_distance(leftEye,rightEye)

		if ear>thresh_eye:
			flag_eye=0
			cv2.putText(frame, "AWAKE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
		else:
			flag_eye+=1
			if flag_eye>=frame_check_mouth:
				cv2.putText(frame, "DROWSINESS DETECTED!!!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
				mixer.music.load("music.wav")
				mixer.music.play()

		if mouth_dist<thresh_mouth:
			cv2.putText(frame, "ACTIVE", (340, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
		else:
			flag_mouth+=1
			if flag_mouth>=60:
				cv2.putText(frame, "YAWNS DETECTED!!!", (290, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
				mixer.music.load("music.wav")
				mixer.music.play()
				flag_mouth=0


		if gaze_direction < 45:
			flag_direction+=1
			if flag_direction>frame_check_direction:
				flag_direction=0
				cv2.putText(frame, "FOCUS ON THE DRIVING", (170, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
				print("focus")
				mixer.music.load("focus.wav")
				mixer.music.play()

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
cv2.destroyAllWindows()
cap.release()