import cv2


class CameraHandler:
    def __init__(self):
        cap = cv2.VideoCapture(0)
