import cv2
import math
import sys
import numpy as np
import time
import pyautogui


def eyesController():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    start = 0
    while True:

        dis = 0

        ret, img = video_capture.read()
        rows, cols, _ = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        Face = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in Face:
            p1face = np.array([x, y])
            p2face = np.array([x + w, y])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            roi_gray = gray[y:y + int(h / 2), x:x + w]
            roi_color = img[y:y + int(h / 2), x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for i, (ex, ey, ew, eh) in enumerate(eyes):
                # disbetHandEh.append(np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1))
                p1 = np.array([0, roi_color.shape[0]])
                p2 = np.array([roi_color.shape[1] + h, roi_color.shape[0]])
                cv2.line(roi_color, (int(p2[0]), int(p2[1])), (int(p1[0]), int(p1[1])), (0, 255, 0), 4)

                p3 = np.array([(int((ex + (ex + ew)) / 2), int((ey + (ey + eh)) / 2))])

                dis=np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)

                cv2.circle(roi_color, (int((ex + (ex + ew)) / 2), int((ey + (ey + eh)) / 2)), 2, (0, 0, 255), 2)

            if len(eyes) == 2:
                start = 0
                ex1 = eyes[0][0] + int(eyes[0][2] / 2)
                ey1 = eyes[0][1] + int(eyes[0][3] / 2)
                ex2 = eyes[1][0] + int(eyes[1][2] / 2)
                ey2 = eyes[1][1] + int(eyes[1][3] / 2)

                # Difference in y coordinates
                dist = math.sqrt(((ex2 - ey2) ** 2) + ((ex1 - ey1) ** 2))
                # print(dis[0] / dist)

                if dis / dist > 0.375:
                    pyautogui.scroll(60)
                    print("up")
                elif dis / dist < 0.24:
                    pyautogui.scroll(-60)
                    print("down")
            elif len(eyes) == 1:
                # print(start-time.time())
                if start == 0:
                    start = time.time()
                elif time.time() - start > 0.5:
                    print('screenshot')
                    image = pyautogui.screenshot()
                    image = cv2.cvtColor(np.array(image),
                                         cv2.COLOR_RGB2BGR)
                    cv2.imwrite("screenShot.png", image)
                    start = 0
            else:
                start = 0
                print(len(eyes))

        # cv2.imshow('Face and Eye Detected', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()

def printing():
    for i in range(1, 1000000):
        print(i)