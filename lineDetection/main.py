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

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([179, 30, 255])
    mask_white = cv2.inRange(HSV_frame, lower_white, upper_white)
    res_white = cv2.bitwise_and(cropped_frame, cropped_frame, mask=mask_white)

    lower_yellow = np.array([20, 70, 100])
    upper_yellow = np.array([30, 255, 255])
    mask_yellow = cv2.inRange(HSV_frame, lower_yellow, upper_yellow)
    res_yellow = cv2.bitwise_and(cropped_frame, cropped_frame, mask=mask_yellow)

    lower_red1 = np.array([0, 70, 100])
    upper_red1 = np.array([10, 255, 255])
    mask_red1 = cv2.inRange(HSV_frame, lower_red1, upper_red1)
    lower_red2 = np.array([170, 70, 100])
    upper_red2 = np.array([179, 255, 255])
    mask_red2 = cv2.inRange(HSV_frame, lower_red2, upper_red2)
    res_red1 = cv2.bitwise_and(cropped_frame, cropped_frame, mask=mask_red1)
    res_red2 = cv2.bitwise_and(cropped_frame, cropped_frame, mask=mask_red2)
    res_red = res_red1 + res_red2

    res_white_yellow_red = res_white + res_yellow + res_red

    # cv2.imshow('frame', frame)
    # cv2.imshow('cropped_frame', cropped_frame)
    # cv2.imshow('white', res_white)
    # cv2.imshow('yellow', res_yellow)
    # cv2.imshow('red', res_red)
    # cv2.imshow('frame', cropped_frame)
    cv2.imshow('res_white_yellow_red', res_white_yellow_red)

    if cv2.waitKey(1) == ord('q'):
        # out = cv2.imwrite('capture_white.jpg', res_white)
        # out = cv2.imwrite('capture_yellow.jpg', res_yellow)
        # out = cv2.imwrite('capture_red.jpg', res_red)
        # out = cv2.imwrite('capture_original.jpg', cropped_frame)
        out = cv2.imwrite('res_white_yellow_red.jpg', res_white_yellow_red)
        break

cap.release()
cv2.destroyAllWindows()

