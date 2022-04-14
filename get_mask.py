from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
#load our model created by trainer.py
"""
maskNet is for the mask detection
faceNEt is for the face detection
"""
maskNet = load_model("mask_detector.model")
prototxtPath = "./face_detector/deploy.prototxt"
caffemodelPath = "./face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, caffemodelPath)

def locMask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),(104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faces = []
    locs = []
    preds = []
    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            
            """
            Draw boxes around detected faces
            """
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h]) 
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            #convert face to rgb
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            
            #append face and box locations to lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))
    #only make a predictions if at least one face was detected
    if len(faces) > 0: 
        #make mask predictions
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
        
    return (locs, preds)

    
#initialize the video stream
#src=0 for webcam
print("Video stream...")
stream = VideoStream(src=0).start()

#loop through frames
while True:
    #get frame and resize
    frame = stream.read()
    frame = imutils.resize(frame, width=1000)
    
    """
    Now we can locate masks and faces then return mask location (loc, preds = This returns how much the model thinks the person is wearing a mask)
    """
    (locs, preds) = locMask(frame, faceNet, maskNet)
    
    """locs will return us with rectangle positions x1,y1,x2,y2 - we will use this to draw rectangles on the faces"""
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        
        #check if mask is on or off
        (mask, noMask) = pred
        label = None
        if mask > noMask:
            label = "Mask"
        elif mask == noMask:
            label = "Mask"
        else:
            label = "No Mask"
        #choose color based on label
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        #get probability
        label = "{}: {:.2f}%".format(label, max(mask, noMask) * 100)
        
        #add text and rectangle to frame
        cv2.putText(frame, label, (startX, endY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        
cv2.destroyAllWindows()
vid.stop()