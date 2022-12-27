# 라이브러리
import tensorflow as tf
import cv2
import numpy as np
import mediapipe as mp

# 모델 불러오기
model = tf.keras.models.load_model('keras_model2.h5')

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 캠 설정
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
str = ""

# 전처리과정
def preprocess(frame,x,y):
    #frame = frame[y-h/2:y+h/2,x-h/2:x+h/2]
    frame_resized = cv2.resize(frame, (224,224), interpolation=cv2.INTER_AREA)
    frame_normalized = (frame_resized.astype(np.float32) / 127.0) - 1
    frame = frame_normalized.reshape((1, 224, 224, 3))
    print(frame.shape)
    
    return frame

# 모델 예측
def predict(pre_frame):
    result = model.predict(pre_frame)
    return result 

# 캠
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cv2.waitKey(33) < 0:
        ret, frame = capture.read()
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = face_detection.process(frame)

        # 영상에 얼굴 감지 주석 그리기 기본값 : True.
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if results.detections:
            x = int(640*results.detections[0].location_data.relative_keypoints[2].x)
            y = int(480*results.detections[0].location_data.relative_keypoints[2].y)
            #h = int(640*results.detections[0].location_data.relative_bounding_box.height)
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)

        pre_frame = preprocess(frame,x,y)
        results = predict(pre_frame)
        predition = model.predict(pre_frame)
        if (predition[0,1] > predition[0,0]):
            str = "Welcome!"
        else:
            str = "No Entry" 
        cv2.putText(frame, str,(20,20),cv2.FONT_ITALIC,1,(255,255,0))
        cv2.imshow("VideoFrame", frame)

capture.release()
cv2.destroyAllWindows()
