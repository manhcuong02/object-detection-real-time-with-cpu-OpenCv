import cv2 as cv
import numpy as np
import time

class Detector:
    def __init__(self, config_path, model_path, classes_path):
        self.config_path = config_path
        self.model_path = model_path
        self.classes_path = classes_path
        
        self.model = cv.dnn_DetectionModel(self.model_path, self.config_path)
        self.model.setInputSize(320,320)
        self.model.setInputScale(1/127.5)
        self.model.setInputMean((127.5, 127.5, 127.5))
        self.model.setInputSwapRB(True)
        self.readClass()
        
        
    def readClass(self):
        with open(self.classes_path, 'r') as f:
            self.classes_list = f.read().splitlines()
                        
    def infer(self, image):
        cls_label, conf,bboxes =  self.model.detect(image, confThreshold = 0.5)
        bboxes = list(bboxes)
        conf = list(np.array(conf).reshape(1,-1)[0])
        conf = list(map(float, conf))
        
        bboxIdx = cv.dnn.NMSBoxes(bboxes, conf, score_threshold = 0.5, nms_threshold = 0.2)
        if len(bboxIdx):
            
            for i in range(0, len(bboxIdx)):
                idx = np.squeeze(bboxIdx[i])
                bbox = bboxes[idx]
                cls_conf = conf[idx]
                label_idx = np.squeeze(cls_label[idx])
                label = self.classes_list[label_idx-1]
                
                x,y,w,h = bbox
                
                text_label = f'{label}: {cls_conf:.2f} '
                image = cv.rectangle(image, (x,y), (x + w, y + h), color = (0,255,0), thickness = 1)
                image = cv.putText(image, text_label, (x,y-10), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)
        return image
        