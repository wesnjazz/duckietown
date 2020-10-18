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

    sensitivity = 120
    lower_white = np.array([0, 0, 255-sensitivity])
    upper_white = np.array([255, sensitivity, 255])
    mask_white = cv2.inRange(HSV_frame, lower_white, upper_white)
    res_white = cv2.bitwise_and(cropped_frame, cropped_frame, mask=mask_white)

    # cv2.imshow('frame', frame)
    # cv2.imshow('cropped_frame', cropped_frame)
    cv2.imshow('frame', res_white)


    if cv2.waitKey(1) == ord('q'):
        out = cv2.imwrite('capture.jpg', HSV_frame)
        break

cap.release()
cv2.destroyAllWindows()

