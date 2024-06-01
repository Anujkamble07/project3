# # import cv2
# # import numpy as np
# # # import mediapipe as mp

# # # INITIALIZING OBJECTS
# # mp_drawing = mp.solutions.drawing_utils
# # mp_drawing_styles = mp.solutions.drawing_styles
# # mp_face_mesh = mp.solutions.face_mesh

# # drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
# # cap = cv2.VideoCapture(0)

# # # DETECT THE FACE LANDMARKS
# # with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
# #   while True:
# #     success, image = cap.read()

# #     # Flip the image horizontally and convert the color space from BGR to RGB
# #     image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

# #     # To improve performance
# #     image.flags.writeable = False
    
# #     # Detect the face landmarks
# #     results = face_mesh.process(image)

# #     # To improve performance
# #     image.flags.writeable = True

# #     # Convert back to the BGR color space
# #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
# #     # Draw the face mesh annotations on the image.
# #     if results.multi_face_landmarks:
# #       for face_landmarks in results.multi_face_landmarks:
# #         mp_drawing.draw_landmarks(
# #             image=image,
# #             landmark_list=face_landmarks,
# #             connections=mp_face_mesh.FACEMESH_TESSELATION,
# #             landmark_drawing_spec=None,
# #             connection_drawing_spec=mp_drawing_styles
# #             .get_default_face_mesh_tesselation_style())

# #     # Display the image
# #     cv2.imshow('MediaPipe FaceMesh', image)
    
# #     # Terminate the process
# #     if cv2.waitKey(5) & 0xFF == 27:
# #       break

# # cap.release()

# # import cv2
# # import dlib

# # cap = cv2.VideoCapture(0)
# # hog_face_detector=dlib.get_frontal_face_detector()
# # dlib_facelandmark =dlib.shape_predictor("C:\\Users\\RUSHAB\\Desktop\\python\\shape_predictor_68_face_landmarks.dat")

# # while True:
# #     frame=cap.read()
# #     gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

# #     faces=hog_face_detector(gray)
# #     for face in faces:
# #         face_landmarks=dlib_facelandmark(gray,face)
# #         for n in range (0,68):
# #             x= face_landmarks.part(n).x
# #             y=face_landmarks.part(n).y
# #             cv2.circle(frame,(x,y),1,(0,255,255),1)
    
# #     cv2.imshow("face landmark",frame)

# #     key=cv2.waitKey(1)
# #     if key ==27:
# #         break
# # cap.release()
# # cv2.destroyAllWindows()

# # from mtcnn.mtcnn import MTCNN
# # import cv2
# # import tensorflow as tf
# # import cv2
# # import numpy as np
# # imagePath=r"C:\Users\RUSHAB\Desktop\photo3.jpg"
# # detector=MTCNN()
# # img=cv2.imread(imagePath)
# # detections=detector.detect_faces(img)

# # # for detection in detections:
# # #     detections=detector.detect_faces(img)

# # for detection in detections:
# #     x,y,w,h=detection["box"]
# #     detected_face=img[int(y):int(y+h),int(x):int(x+w)]
# #     cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
# #     keypoints=detection["keypoints"]
# #     left_eye=list(keypoints["left_eye"])
# #     right_eye=list(keypoints["right_eye"])
# #     left_mouth=list(keypoints["mouth_left"])
# #     right_mouth=list(keypoints["mouth_right"])
# #     nose=list(keypoints["nose"])
# #     cv2.circle(img,left_eye,2,color=(0,255,255),thickness=2)
# #     cv2.circle(img,right_eye,2,color=(0,255,255),thickness=2)
# #     cv2.circle(img,left_mouth,2,color=(0,255,255),thickness=2)
# #     cv2.circle(img,right_mouth,2,color=(0,255,255),thickness=2)
# #     cv2.circle(img,nose,2,color=(0,255,255),thickness=2)
# # cv2.imshow("landmark found",img)
# # cv2.waitKey(0)        

# from imutils import face_utils
# import maths


# import numpy as np

# import argparse

# import imutils

# import dlib

# import cv2

# ap = argparse.ArgumentParser()

# # –-image is a path to the input image

# ap.add_argument("p", "–shape-predictor", required=True,

# help="path to facial landmark predictor")

# ap.add_argument("-i", "–image", required=True,

# help="path to input image")

# args = vars(ap.parse_args())

# # initialize built-in face detector in dlib

# detector = dlib.get_frontal_face_detector()

# # initialize face landmark predictor

# predictor = dlib.shape_predictor(args["shape_predictor"])

# # load input image, resize it, and convert it to grayscale

# image = cv2.imread(args["image"])

# image = imutils.resize(image, width=500)

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # detect faces in the grayscale image

# rects = detector(gray, 1)    

# for (i, rect) in enumerate(rects):

# # predict facial landmarks in image and convert to NumPy array

# shape = predictor(gray, rect)

# shape = face_utils.shape_to_np(shape)

# # convert to OpenCV-style bounding box

# cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# # show the face number and draw facial landmarks on the image

# cv2.putText(image, "Face #{}".format(i + 1), (x – 10, y – 10),

# cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# for (x, y) in shape:

# cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

# # show the resulting output image

# cv2.imshow(“Output”, image)

# cv2.waitKey(0)
  

import face_recognition
import PIL.Image
import PIL.ImageDraw

input_image=face_recognition.load_image_file("C:\\Users\\RUSHAB\\Desktop\\4.jpg")
face_landmarks=face_recognition.face_landmarks(input_image)
print(face_landmarks[0])

output_image=PIL.Image.fromarray(input_image)
draw=PIL.ImageDraw.Draw(output_image)
landmark=face_landmarks[0].get('right_eye')

# landmark=face_landmarks[0].get('nose_tip')

for loc in landmark:
    x,y=loc
    draw.rectangle((x,y,x+2,y+2),outline='green')
output_image.show(  )