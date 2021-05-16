import cv2
import numpy as np
from constants import *
import imutils
from imutils.video import VideoStream
from pygame import mixer
import threading
import os


class Rescue_PI:
    run_program = RUN_PROGRAM
    input_video_file_path = None
    preferable_target = cv2.dnn.DNN_TARGET_CPU

    def __init__(self):
        self.SoundThread = None
        self.AudioPlay = None
        self.rescue_model = None
        self.frame = None
        self.h = None
        self.w = None
        self.vs = None
        self.image_blob = None
        self.confidence = None
        self.detections = None
        self.box = None
        self.human_blob = None
        self.f_h = None
        self.f_w = None
        self.startX = None
        self.startY = None
        self.endX = None
        self.endY = None
        self.human_blob = None
        self.predictions = None
        self.name = None
        self.detector = None
        self.prediction_index = None
        self.fileName = None
        self.text = None
        self.y = None
        self.colorIndex = None
        self.threshold = MIN_THRESHOLD
        self.model_input_size = MODEL_INPUT_SIZE
        self.current_time = None
        self.time = ""
        self.seconds = None
        self.debug = False
        self.sound_thread = None
        self.use_graphics = USE_GRAPHICS
        self.voice = None
        self.sound = None
        self.idx = None
        self.label = None
        self.classes = CLASSES

        self.load_caffe_model()
        self.load_onnx_model()
        self.init_audio()
        self.create_play_audio_thread()
        self.initialize_camera()

    @classmethod
    def perform_job(cls, preferableTarget=preferable_target, input_video_file_path=input_video_file_path):
        """
        This method performs the job expected from this class.
        :key
        """
        # Set preferable target.
        Rescue_PI.preferable_target = preferableTarget
        # Set input video file path (if applicable)
        Rescue_PI.input_video_file_path = input_video_file_path
        # Create a thread that uses the thread_for_mask_detection function and start it.
        t1 = threading.Thread(target=Rescue_PI().thread_for_rescue_detection)
        t1.start()
        # print("[INFO] Starting Process for Mask Detection")
        # p1 = Process(target=Rescue_PI().thread_for_mask_detection)
        # p1.start()
        # t1.join()

    def is_blur(self, frame, thresh):
        fm = cv2.Laplacian(frame, cv2.CV_64F).var()
        if fm < thresh:
            return True
        else:
            return False

    def super_res(self, frame):
        self.frame = cv2.resize(frame, (self.model_input_size, self.model_input_size), interpolation=cv2.INTER_CUBIC)

    def load_caffe_model(self):
        """
        This function will load the caffe model that we will use for detecting a human_blob, and then set the preferable target to the correct target.
        :key
        """
        print("Loading caffe model used for detecting a human_blob.")

        # Use cv2.dnn function to read the caffe model used for detecting faces and set preferable target.
        # self.detector = cv2.dnn.readNetFromCaffe(os.path.join(
        #     os.path.dirname(os.path.realpath(__file__)),
        #     prototxt_path),
        #     os.path.join(
        #         os.path.dirname(os.path.realpath(__file__)),
        #         human_model_path))
        self.detector = cv2.dnn.readNetFromCaffe(prototxt="human_detection_model/MobileNetSSD_deploy.prototxt.txt",
                                                 caffeModel="human_detection_model/MobileNetSSD_deploy.caffemodel")
        self.detector.setPreferableTarget(Rescue_PI.preferable_target)

    def load_onnx_model(self):
        """
        This function will load the pytorch model that is used for predicting the class of the human_blob.
        :key
        """
        print("Loading Rescue Detection Model")

        self.rescue_model = cv2.dnn.readNetFromONNX(os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            rescue_cnn_model_path))

        self.rescue_model.setPreferableTarget(Rescue_PI.preferable_target)

    def initialize_camera(self):
        """
        This function will initialize the camera or video stream by figuring out whether to stream the camera capture or from a video file.
        :key
        """
        if Rescue_PI.input_video_file_path is None:
            print("[INFO] starting threaded video stream...")
            self.vs = VideoStream(src=VID_CAM_INDEX).start()
        else:
            self.vs = cv2.VideoCapture(Rescue_PI.input_video_file_path)

    def grab_next_frame(self):
        """
        This function extracts the next frame from the video stream.
        :return:
        """
        if Rescue_PI.input_video_file_path is None:
            self.orig_frame = self.vs.read()
            self.frame = self.orig_frame.copy()
        else:
            _, self.frame = self.vs.read()
        # self.frame = cv2.rotate(self.frame, cv2.ROTATE_180)
        if self.frame is None:
            pass
        else:
            self.frame = imutils.resize(self.frame, width=frame_width_in_pixels)

    def set_dimensions_for_frame(self):
        """
        This function will set the frame dimensions, which we will use later on.
        :key
        """
        if not self.h or not self.w:
            (self.h, self.w) = self.frame.shape[:2]

    def create_frame_blob(self):
        """
        This function will create a blob for our human_blob detector to detect a human_blob.
        :key
        """
        # self.image_blob = cv2.dnn.blobFromImage(
        #     cv2.resize(self.frame, (300, 300)), 1.0, (300, 300),
        #     (104.0, 177.0, 123.0), swapRB=False, crop=False)
        self.image_blob = cv2.dnn.blobFromImage(cv2.resize(self.frame, (300, 300)),
                                                0.007843, (300, 300), 127.5)

    def extract_face_detections(self):
        """
        This function will extract each human_blob detection that our human_blob detection model provides.
        :return:
        """
        self.detector.setInput(self.image_blob)
        self.detections = self.detector.forward()

    def extract_confidence_from_human_detections(self, i):
        """
        This function will extract the confidence(probability) of the human_blob detection so that we can filter out weak detections.
        :param i:
        :return:
        """
        self.confidence = self.detections[0, 0, i, 2]

    def get_class_label(self, i):
        self.idx = int(self.detections[0, 0, i, 1])
        self.label = round(self.idx)

    def create_human_box(self, i):
        """
        This function will define coordinates of the human_blob.
        :param i:
        :return:
        """
        self.box = self.detections[0, 0, i, 3:7] * np.array([self.w, self.h, self.w, self.h])
        (self.startX, self.startY, self.endX, self.endY) = self.box.astype("int")

    def extract_human_roi(self):
        """
        This function will use the coordinates defined earlier and create a ROI that we will use for embeddings.
        :return:
        """
        self.human_blob = self.frame[self.startY:self.endY, self.startX:self.endX]
        (self.f_h, self.f_w) = self.human_blob.shape[:2]

    def create_predictions_blob(self):
        """
        This function will create another blob out of the human_blob ROI that we will use for prediction.
        :return:
        """
        self.human_blob = cv2.dnn.blobFromImage(cv2.resize(self.human_blob,
                                                           (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)), 1.0 / 255,
                                                (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), (0, 0, 0),
                                                swapRB=True, crop=False)

    def extract_detections(self):
        """
        This function uses the PyTorch model to predict from the given human_blob blob.
        :return:
        """
        self.rescue_model.setInput(self.human_blob)
        self.predictions = self.rescue_model.forward()

    def perform_classification(self):
        """
        This function will now use the prediction to do the following:
            1. Extract the class prediction from the predictions.
            2. Get the label of the prediction.
        :return:
        """
        self.prediction_index = np.array(self.predictions)[0].argmax()
        print(self.prediction_index)
        if self.prediction_index == FIGHTING_INDEX:
            self.name = "Fighting"
        elif self.prediction_index == CRYING_INDEX:
            self.name = "Crying"
        elif self.prediction_index == NORMAL_INDEX:
            self.name = "Normal"
        else:
            pass

    def init_audio(self):
        mixer.init()
        mixer.set_num_channels(8)
        self.voice = mixer.Channel(5)
        self.sound = mixer.Sound(sound_file)

    def play_audio(self):
        """
        This function is used for playing the alarm if a person is not wearing a mask.
        :return:
        """
        if not self.voice.get_busy():
            self.voice.play(self.sound)
        else:
            pass

    def create_play_audio_thread(self):
        """
        This function is used for creating a thread for the audio playing so that there won't be a blocking call.
        """
        self.sound_thread = threading.Thread(target=self.play_audio)

    def create_frame_icons(self):
        """
        This function will create the icons that will be displayed on the frame.
        :return:
        """
        self.text = "{}".format(self.name)
        self.y = self.startY - 10 if self.startY - 10 > 10 else self.startY + 10
        self.colorIndex = LABELS.index(self.name)

    def loop_over_frames(self):
        """
        This is the main function that will loop through the frames and use the functions defined above to detect for human_blob mask.
        :return:
        """
        while Rescue_PI.run_program:
            self.grab_next_frame()
            self.set_dimensions_for_frame()
            self.create_frame_blob()
            self.extract_face_detections()
            for i in range(0, self.detections.shape[2]):
                self.extract_confidence_from_human_detections(i)
                if self.confidence > MIN_CONFIDENCE:
                    self.get_class_label(i)
                    if self.label == 15:
                        self.create_human_box(i)
                        self.extract_human_roi()
                        if self.f_w < 20 or self.f_h < 20:
                            continue
                        if self.is_blur(self.human_blob, self.threshold):
                            continue
                        else:
                            self.super_res(self.human_blob)
                            self.create_predictions_blob()
                            self.extract_detections()
                            self.perform_classification()
                            if self.name == "Fighting":
                                print("[Prediction] Fighting is occurring")
                                self.play_audio()
                            if self.name == "Crying":
                                print("[Prediction] Crying is occurring")
                                self.play_audio()
                            if self.name == "Normal":
                                print("[Prediction] Normal")
                            if self.use_graphics:
                                self.create_frame_icons()
                                cv2.putText(self.orig_frame, self.text, (15, 15), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.45, COLORS[self.colorIndex], 2)
                    else:
                        pass
            if OPEN_DISPLAY:
                cv2.imshow("Frame", self.orig_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break

    def clean_up(self):
        """
        Clean up the cv2 video capture.
        :return:
        """
        cv2.destroyAllWindows()
        # self.vs.release()

    def thread_for_rescue_detection(self):
        """
        Callable function that will run the mask detector and can be invoked in a thread.
        :return:
        """
        try:
            self.loop_over_frames()
        except Exception as e:
            pass
        finally:
            self.clean_up()


if __name__ == "__main__":
    Rescue_PI.perform_job(preferableTarget=cv2.dnn.DNN_TARGET_MYRIAD)
