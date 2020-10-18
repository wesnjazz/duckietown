import numpy as np
import cv2 as cv2


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame_height, frame_width, frame_channel = frame.shape

    # ROI region
    ROI_left_upper = (0, (frame_height//100)*55)
    ROI_right_bottom = (frame_width, frame_height)
    ROI_border_color = (0, 0, 255)
    ROI_border_thickness = 3

    # Frames
    # ROI_drawn_frame = cv2.rectangle(frame, ROI_left_upper, ROI_right_bottom, ROI_border_color, ROI_border_thickness)
    cropped_frame = frame[ROI_left_upper[1]:, :, :]
    HSV_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)

    # cv2.imshow('frame', frame)
    # cv2.imshow('cropped_frame', cropped_frame)
    cv2.imshow('frame', HSV_frame)

    if cv2.waitKey(1) == ord('c'):
        out = cv2.imwrite('capture.jpg', HSV_frame)
        break
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

