import cv2
from cv2 import CascadeClassifier
import time
from config import config
from client_model import ClientModelInterface
import matplotlib.pyplot as plt
import imutils
import numpy as np

input_path = '/home/wuyang/datasets/2ndFAmstrdm-2/L2ndFloor-D2020-06-17_T14-00-01.ts'
classifier = CascadeClassifier('haarcascade_frontalface_default.xml')

app = ClientModelInterface(config, client=False)

objs = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
        'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
        'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
        'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
        'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')


def detect_face(frame):
    return classifier.detectMultiScale(frame)


def mask(frame, bbox):
    bbox = bbox.astype(int)
    for (x1, y1, x2, y2) in bbox:
        dy = y2 - y1
        dx = (x2 - x1)
        x1 = int(x1 + dx * 0.25)
        x2 = int(x2 - dx * 0.25)
        y2 = int(y1 + dy * 0.2)
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
        roi_color = frame[y1:y2, x1:x2]
        # blur the colored image
        blur = cv2.GaussianBlur(roi_color, (5, 5), 0)
        # Insert ROI back into image
        frame[y1:y2, x1:x2] = blur
        # frame[x1:x2, y1:y2] = blur


def draw_box(frame, bbox):
    bbox = bbox.astype(int)
    for x, y, x2, y2 in bbox:
        cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 2)


def show(frame, title=None):
    plt.imshow(frame)
    if title is not None:
        plt.title(title)
    plt.show()


def stretch(img):
    max = float(img.max())
    min = float(img.min())

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = (255 / (max - min)) * img[i, j] - (255 * min) / (max - min)
    return img


def detect_plate(frame, bbox, index=0):
    bbox = bbox.astype(int)
    car_frame = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    a = 400 * car_frame.shape[0] / car_frame.shape[1]
    a = int(a)
    img = cv2.resize(car_frame, (400, a))
    height, width = img.shape[:2]

    height_ratio = img.shape[0] / car_frame.shape[0]
    width_ratio = img.shape[1] / car_frame.shape[1]

    # cv2.imwrite('./car_{}.png'.format(index), car_frame)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.bilateralFilter(gray, 11, 17, 17)
    # stretchedimg = stretch(gray)
    stretchedimg = gray
    cannyimg = cv2.Canny(stretchedimg, stretchedimg.shape[0], stretchedimg.shape[1])

    kernel = np.ones((5, 5), np.uint8)
    closingimg = cv2.morphologyEx(cannyimg, cv2.MORPH_CLOSE, kernel)

    openingimg = cv2.morphologyEx(closingimg, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((5, 5), np.uint8)
    openingimg = cv2.morphologyEx(openingimg, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(openingimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for c in contours:
        # peri = cv2.arcLength(c, True)
        # approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if len(approx) == 4:
        #     print(approx)
        x1 = np.min(c[:, :, 0])
        y1 = np.min(c[:, :, 1])
        x2 = np.max(c[:, :, 0])
        y2 = np.max(c[:, :, 1])

        if x2 - x1 > width * 0.1 or y2 - y1 > height * 0.1:
            continue
        # img[y1: y2, x1: x2] = (0, 255, 0)
        # cv2.imwrite('output_index{}.png'.format(index), img)

        x1 = int(x1 / width_ratio)
        x2 = int(x2 / width_ratio)
        y1 = int(y1 / height_ratio)
        y2 = int(y2 / height_ratio)

        if x2 - x1 < 1 or y2 - y1 < 1:
            continue

        roi = frame[bbox[1] + y1: bbox[1] + y2, bbox[0] + x1: bbox[0] + x2]

        blur = cv2.GaussianBlur(roi, (21, 21), 0)
        frame[bbox[1] + y1: bbox[1] + y2, bbox[0] + x1: bbox[0] + x2] = blur

    # for c in contours:
    #     # approximate the contour
    #     peri = cv2.arcLength(c, True)
    #     approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    #     # if our approximated contour has four points, then
    #     # we can assume that we have found our screen
    #     if len(approx) == 4:
    #         x1 = np.min(approx[:, :, 0])
    #         x2 = np.max(approx[:, :, 0])
    #         y1 = np.min(approx[:, :, 1])
    #         y2 = np.max(approx[:, :, 1])
    #         if x2 - x1 > width * 0.2 or y2 - y1 > height * 0.2 or x2 - x1 < 3:
    #             continue
    #         print(x1, y1)
    #         # x1, y1 = c[:2]
    #         # x2 = x1 + c[2]
    #         # y2 = x2 + c[3]
    #         roi_color = car_frame[y1:y2, x1:x2]
    #         if roi_color is not None:
    #             blur = cv2.GaussianBlur(roi_color, (101, 101), 0)
    #             #frame[bbox[0] + x1: bbox[0] + x2, bbox[1] + y1, bbox[1] + y2] = (255, 255, 0)
    #             print('y coor', bbox[1] + y1, bbox[1] + y2)
    #             print('x coor', bbox[0] + x1, bbox[0] + x2)
    #             # tmp = car_frame.copy()
    #             # tmp[y1:y2, x1:x2] = (0, 255, 0)
    #             # show(tmp, 'x {}, y {}'.format(x1, y1))
    #             frame[bbox[1] + y1: bbox[1] + y2, bbox[0] + x1: bbox[0] + x2] = (0, 255, 0)
    #             #frame[bbox[1] + x1: bbox[1] + x2, bbox[0] + y1: bbox[0] + y2] = (0, 255, 0)


def proc_frame(path):
    frame = cv2.imread(path)
    f = frame.copy()
    result = app.run(frame)
    person = result[0][0][:, :4]
    # draw_box(frame, person)

    car = result[0][2][:, :4]
    # draw_box(frame, car)
    # convert to gray scale

    index = 0
    for c in car:
        detect_plate(frame, c, index)
        index += 1

    cv2.imwrite('./output.png', frame)
    # cv2.imshow('frame', frame)
    # cv2.waitKey(1)


def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    i = 0
    fps = cap.get(cv2.CAP_PROP_FPS)

    _, frame = cap.read()

    scale = 0.5
    width, height = int(frame.shape[1] * scale), int(frame.shape[0] * scale)
    output = cv2.VideoWriter('filename3.avi',
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        start = time.time()
        i += 1
        # if i % 30 != 0:
        #     continue
        frame = cv2.resize(frame, (width, height))
        raw = frame.copy()
        # person and face detection
        result = app.run(frame)
        person = result[0][0][:, :4]
        # draw_box(frame, person)
        mask(frame, person)

        # vehicle & plate detection
        car = result[0][2][:, :4]

        for c in car:
            detect_plate(frame, c)

        # output.write(frame)

        # frame = app.model.show_result(frame, result)
        print('index {} runtime {} s'.format(i, time.time() - start))

        # scale_percent = 50  # percent of original size
        # width = int(frame.shape[1] * scale_percent / 100)
        # height = int(frame.shape[0] * scale_percent / 100)
        # dim = (width, height)
        #
        # frame = cv2.resize(frame, dim)
        cv2.imshow('raw', raw)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(3000)
        if key == ord('s'):
            cv2.imwrite('raw_{}.png'.format(i), raw)
            cv2.imwrite('frame_{}.png'.format(i), frame)
        else:
            continue


def test(index):
    import cv2
    import numpy as np
    from math import sqrt
    import uuid

    boxes = []

    # param contains the center and the color of the circle
    def draw_red_circle(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            boxes.append((x, y))

        elif event == cv2.EVENT_LBUTTONUP:
            crop = raw[boxes[-1][1]:y, boxes[-1][0]:x]
            frame_crop = frame[boxes[-1][1]:y, boxes[-1][0]:x]
            scale = 2
            width, height = int(crop.shape[1] * scale), int(crop.shape[0] * scale)
            output = cv2.resize(crop, (width, height), interpolation=cv2.INTER_NEAREST)
            frame_output = cv2.resize(frame_crop, (width, height), interpolation=cv2.INTER_NEAREST)

            print(output.shape, frame_output.shape)

            out = cv2.hconcat([output, frame_output])
            index = uuid.uuid1().__str__().split('-')[0]
            cv2.imwrite('compare_{}.png'.format(index), out)
            # cv2.imshow('resize', out)
            # cv2.waitKey(10000)
            # cv2.rectangle(img, boxes[-1], (x, y), 1)

    raw = cv2.imread('raw_{}.png'.format(index))
    frame = cv2.imread('frame_{}.png'.format(index))
    cv2.namedWindow("img_red")

    param = [(200, 200), (0, 0, 255)]
    cv2.setMouseCallback("img_red", draw_red_circle, param)

    while True:
        cv2.imshow("img_red", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()

import glob
for file in glob.glob('raw_*.png'):
    index = file.split('_')[1].split('.')[0]
    test(index)
# zoom(177)
# proc_frame(path)
