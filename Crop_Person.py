import time
import cv2
import imutils
import numpy as np
import os
import datetime
import glob
import shutil


class Crop_and_Save:
    def __init__(self):
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]
        self.net = None
        self.prototxt_path = '/home/srinivassriram/Desktop/Rescue_Pi_Code/Rescue_Pi_Code/human_detection_model' \
                             '/MobileNetSSD_deploy.prototxt'
        self.model_path = '/home/srinivassriram/Desktop/Rescue_Pi_Code/Rescue_Pi_Code/human_detection_model' \
                          '/MobileNetSSD_deploy.caffemodel'
        self.load_caffe_model()
        self.files = []
        self.images_path = '/home/srinivassriram/Desktop/Rescue_Pi_Code/Rescue_Pi_Code/Dataset/Normal/'
        self.init_files(self.images_path)
        self.img = None
        self.h = None
        self.w = None
        self.image_blob = None
        self.detections = None
        self.confidence = None
        self.idx = None
        self.label = None
        self.box = None
        self.w = None
        self.h = None

    def load_caffe_model(self):
        self.net = cv2.dnn.readNetFromCaffe(prototxt="human_detection_model/MobileNetSSD_deploy.prototxt",
                                            caffeModel="human_detection_model/MobileNetSSD_deploy.caffemodel")

    def init_files(self, path):
        for f in os.listdir(path):
            self.files.append(self.images_path + f)

    def read_image(self, path):
        self.img = cv2.imread(path)
        self.img = cv2.resize(self.img, (640, 480), interpolation = cv2.INTER_CUBIC)
        cv2.imshow("Image", self.img)
        cv2.waitKey(0)

    def set_dimensions_for_frame(self):
        """
        This function will set the frame dimensions, which we will use later on.
        :key
        """
        if not self.h or not self.w:
            (self.h, self.w) = self.img.shape[:2]

    def create_frame_blob(self):
        """
        This function will create a blob for our human_blob detector to detect a human_blob.
        :key
        """
        self.image_blob = cv2.dnn.blobFromImage(
            cv2.resize(self.img, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)
        print(self.image_blob)

    def extract_face_detections(self):
        """
        This function will extract each human_blob detection that our human_blob detection model provides.
        :return:
        """
        self.net.setInput(self.image_blob)
        self.detections = self.net.forward()

    def extract_confidence_from_face_detections(self, i):
        """
        This function will extract the confidence(probability) of the human_blob detection so that we can filter out weak detections.
        :param i:
        :return:
        """
        self.confidence = self.detections[0, 0, i, 2]

    def extract_label(self, i):
        self.idx = int(self.detections[0, 0, i, 1])
        self.label = round(self.idx)

    def extract_coordinates(self, i):
        self.box = (self.detections[0, 0, i, 3:7] * np.array([self.w, self.h, self.w, self.h])).astype('int')

    def crop_human(self, frame):
        # xCorr, yCoor, width, height = self.box[0], self.box[1], self.box[2] - self.box[0], self.box[3] - self.box[1]
        frame = frame[self.box[0], self.box[1], self.box[2], self.box[3]]
        # cv2.imshow("Cropped Frame", frame)
        # cv2.waitKey(0)
        return frame

    def perform_job(self):
        print(self.files)
        for self.current_file in self.files:
            print(self.current_file)
            self.read_image(self.current_file)
            self.set_dimensions_for_frame()
            self.create_frame_blob()
            self.extract_face_detections()

            for i in np.arange(0, self.detections.shape[2]):
                print("Inside For loop")
                self.extract_confidence_from_face_detections(i)
                print(self.confidence)
                if self.confidence > 0.5:
                    self.extract_label(i)
                    if self.label == 15:
                        print("Detected human")
                        self.extract_face_detections(i)
                        self.img = self.crop_human(self, self.img)
                        print("Saving")
                        cv2.imwrite(self.img, self.current_file)
                    else:
                        pass
                else:
                    pass


if __name__ == "__main__":
    Crop_and_Save_inst = Crop_and_Save()
    Crop_and_Save_inst.perform_job()
