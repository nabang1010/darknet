import numpy as np
import cv2
import darknet
import time

vidPath = "/home/nab/Desktop/nabang1010/OxfordTownCenterDataset/TownCentreXVID.mp4"
weightPath = "/home/nab/Desktop/darknetResource/yolov4.weights"
cfgPath = "/home/nab/Desktop/darknetResource/yolov4.cfg"
dataPath = "/home/nab/Desktop/darknet/cfg/coco.data"
       

def main():
    network, class_names, class_colors = darknet.load_network(cfgPath, dataPath, weightPath,  batch_size=1)
    cap = cv2.VideoCapture(vidPath)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height),
                                        interpolation=cv2.INTER_LINEAR)
        frame_queue.put(frame)
        img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
        darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
        darknet_image_queue.put(img_for_detect)
    cap.release()
if __name__ == '__main__':
    main()