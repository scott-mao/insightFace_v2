from __future__ import print_function

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from retinaface.data import cfg_mnet
from retinaface.layers.functions.prior_box import PriorBox
from retinaface.loader import load_model
from retinaface.utils.box_utils import decode, decode_landm
from retinaface.utils.nms.py_cpu_nms import py_cpu_nms
from glob import glob
import cv2
import time
import os
import multiprocessing
import tqdm
import imutils

input_folder_path = "/home/tupm/HDD/datasets/2d_face/part1/dir_001_filtered/*/*"
output_folder_path = "/home/tupm/HDD/datasets/2d_face/out"

cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
model = load_model().to(device)


# model.eval()

def detect_faces(img_raw, confidence_threshold=0.9, top_k=5000, nms_threshold=0.4, keep_top_k=10, resize=1):
    global model
    img = np.float32(img_raw.copy())
    im_height, im_width = img.shape[:2]
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    # tic = time.time()
    with torch.no_grad():
        loc, conf, landms = model(img)  # forward pass
        # print('net forward time: {:.4f}'.format(time.time() - tic))

    priorbox = PriorBox(cfg_mnet, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg_mnet['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg_mnet['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:keep_top_k, :]
    landms = landms[:keep_top_k, :]
    # print(landms.shape)
    landms = landms.reshape((-1, 5, 2))
    # print(landms.shape)
    landms = landms.transpose((0, 2, 1))
    # print(landms.shape)
    landms = landms.reshape(-1, 10, )
    # print(landms.shape)
    del img
    del scale
    return dets, landms


def get_mask_image(image_path):
    global output_folder_path
    parrent_dir = os.path.basename(os.path.dirname(image_path))
    img_name = os.path.basename(image_path)
    output_folder = os.path.join(output_folder_path, parrent_dir)
    os.makedirs(output_folder, exist_ok=True)

    frame = cv2.imread(image_path)
    t1 = time.time()
    if frame is None:
        return
    if frame.shape[1] > 1024:
        frame = imutils.resize(frame, width=1024)
    dets, landms = detect_faces(frame)
    dets = np.array(dets)
    # print(1/(time.time() - t1), frame.shape,img_name)
    for idx, box in enumerate(dets):
        x1, y1, x2, y2, _ = box
        if x2 - x1 < 100:
            continue
        face = frame[int(y1): int(y2), int(x1): int(x2), :]
        org_face = face.copy()
        landm = landms[idx]
        landm = np.array(landm)
        landm.astype(int)
        landm = landm.reshape((2, 5))
        x = landm[0]
        y = landm[1]
        yc = (y[0] + y[1]) / 2 - y1 + (x[1] - x[0]) / 3
        h, w, _ = face.shape
        rectangle_cnt = np.array([(0, yc), (w, yc), (w, h), (0, h)]).astype(np.int32)
        cv2.drawContours(face, [rectangle_cnt], 0, (0, 0, 0), -1)
        cv2.imwrite(os.path.join(output_folder, 'augmented' + img_name), face)
        cv2.imwrite(os.path.join(output_folder, img_name), org_face)
        del face
        break
    del frame

    # if time.time() - t1 > 0.5:
    #     cv2.imshow('frame', frame)
    #     cv2.waitKey(0)


def test_video():
    video = cv2.VideoCapture('/home/tupm/Videos/20191214_102534.mp4')

    while True:

        ret, frame = video.read()
        t1 = time.time()
        dets, landms = detect_faces(frame)
        dets = np.array(dets)

        for idx, box in enumerate(dets):
            x1, y1, x2, y2, _ = box
            face = frame[int(y1): int(y2), int(x1): int(x2), :]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            landm = landms[idx]
            landm = np.array(landm)
            landm.astype(int)
            landm = landm.reshape((2, 5))
            x = landm[0]
            y = landm[1]
            yc = (y[0] + y[1]) / 2 - y1 + (x[1] - x[0]) / 3
            mask = np.ones(face.shape)
            h, w, _ = face.shape
            rectangle_cnt = np.array([(0, yc), (w, yc), (w, h), (0, h)]).astype(np.int32)
            cv2.drawContours(face, [rectangle_cnt], 0, (0, 0, 0), -1)
            for i in range(2):
                cv2.circle(frame, (x[i], y[i]), 2, (0, 0, 255), 2)
                cv2.circle(face, (x[i] - x1, y[i] - y1), 2, (0, 0, 255), 2)
            cv2.imshow('face', face)
        print(1 / (time.time() - t1))

        cv2.imshow('test', frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    images = glob(input_folder_path)
    # multiprocessing.set_start_method('spawn')
    # pool = multiprocessing.Pool(2)

    # for _ in tqdm.tqdm(pool.imap_unordered(get_mask_image, images)):
    #     pass
    # for i in tqdm.tqdm(range(len(images))):
    #     get_mask_image(images[i])

    test_video()
