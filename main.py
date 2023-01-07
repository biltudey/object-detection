import streamlit as st
import cv2
import numpy as np
import time
from PIL import Image



def draw_bbox(img,id,score,x,y,w,h):
    label = str(classes[id])
    color = COLORS[id]
    cv2.rectangle(img,(x,y),(w,h),color,1)
    cv2.putText(img,label,(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.9, color, 1)


st.title('Object Detection')

uploaded_file = st.file_uploader("Upload Image",['png','jpg','jpeg'])
if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 3)
    # Now do something with the image! For example, let's display it:
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    st.image(img,width=800)
    # st.image(img)









    Width = img.shape[1]
    Height = img.shape[0]
    classes = open('coco.names').read().strip().split('\n')
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))


    net = cv2.dnn.readNetFromDarknet('yolov3.cfg','yolov3.weights')
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)



    blob=cv2.dnn.blobFromImage(img,1/255.0,(416,416),swapRB=True, crop=False)


    net.setInput(blob)
    outputs = net.forward(ln)
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outputs:
        # print(out.shape)
        # print(out)
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores,axis=0)
            confidence = scores[class_id]
            if confidence >0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes,confidences,conf_threshold,nms_threshold)

    if len(indices)>0:
        for i in indices:
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            draw_bbox(img, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))


    st.subheader(f'After Processing the image having {len(indices)} objects.')
    st.image(img,width=800)