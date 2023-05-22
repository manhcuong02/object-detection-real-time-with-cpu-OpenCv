from detect import *
import os
import cv2 as cv
import time

def main():
    video_path = 'data/video.mp4'
    config_path = 'model_data/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    classes_path = 'model_data/coco.names'
    model_path = 'model_data/frozen_inference_graph.pb'
    
    model = Detector(config_path, model_path, classes_path)
    
    cap = cv.VideoCapture(video_path)
    
    while True:
        start = time.time()
        ret, frame = cap.read()
        if ret is False:
            break
        frame = model.infer(frame)
        end = time.time()
        fps = round(1.0/(end - start), 2)
        cv.putText(frame, f"FPS: {fps}", (100, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
        
        cv.imshow('image', frame)
        key = cv.waitKey(1)    
        if key == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
    
if __name__ =='__main__':
    main()
