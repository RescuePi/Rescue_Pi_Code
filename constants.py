RUN_PROGRAM = True

prototxt_path = "human_detection_model/MobileNetSSD_deploy.prototxt.txt"

human_model_path = "human_detection_model/MobileNetSSD_deploy.caffemodel"

rescue_cnn_model_path = "saved_models/PyTorch_Models/Final_Rescue_Model_Onnx.onnx"

sound_file = "alarm.wav"

MIN_CONFIDENCE = 0.8

frame_width_in_pixels = 320

OPEN_DISPLAY = True

USE_VIDEO = True

USE_GRAPHICS = True

VID_CAM_INDEX = 0

MODEL_INPUT_SIZE = 128

SLEEP_TIME_AMOUNT = 2

LABELS = ["Fighting", "Crying", "Normal"]

COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]

MIN_THRESHOLD = 200

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

FIGHTING_INDEX = 0

CRYING_INDEX = 1

NORMAL_INDEX = 2
