import face_alignment
import cv2
from skimage import io
import numpy as np
from math import *

class preprocess_landmarks(object):
    def __init__(self):
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda')

    def detect(self, img_path):
        input = io.imread(img_path)
        preds = self.fa.get_landmarks(input)
        return preds[0]


class preprocess_rotate(object):

    def eye_angle(self, coordinates):
    # 利用左右眼各6个landmarks求取图片摆正所需旋转角度
        eyel_x = int(np.round(
            np.mean([coordinates[36][0], coordinates[37][0], coordinates[38][0], coordinates[39][0], coordinates[40][0],
                     coordinates[41][0]])))
        eyel_y = int(np.round(
            np.mean([coordinates[36][1], coordinates[37][1], coordinates[38][1], coordinates[39][1], coordinates[40][1],
                     coordinates[41][1]])))
        eyer_x = int(np.round(
            np.mean([coordinates[42][0], coordinates[43][0], coordinates[44][0], coordinates[45][0], coordinates[46][0],
                     coordinates[47][0]])))
        eyer_y = int(np.round(
            np.mean([coordinates[42][1], coordinates[43][1], coordinates[44][1], coordinates[45][1], coordinates[46][1],
                     coordinates[47][1]])))
        angle = np.arctan((eyer_y - eyel_y) / (eyer_x - eyel_x)) * 180 / np.pi

        return angle

    def rotate(self, img, angle, coordinates):
    # 对图像旋转angle角度，并将landmarks旋转angle角度
        width = img.shape[1]
        height = img.shape[0]
        heightNew = int(width * fabs(sin(radians(angle))) + height * fabs(cos(radians(angle))))  # 旋转后的高度
        widthNew = int(height * fabs(sin(radians(angle))) + width * fabs(cos(radians(angle))))  # 旋转后的宽度
        MatRotation = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1)  # 旋转矩阵
        rot_img = cv2.warpAffine(img, MatRotation, (widthNew, heightNew), borderValue=(255, 255, 255))  # 旋转后的图像
        rot_coordinates = np.zeros((len(coordinates), 2), dtype='int')
        # 旋转后的坐标
        for idx, coordinate in enumerate(coordinates):
            rot_coordinates[idx] = np.reshape(np.dot(MatRotation, np.array([[coordinate[0]], [coordinate[1]], [1]])),
                                              (1, 2))
        return rot_img, rot_coordinates

class preprocess_crop(object):

    def crop(self, rot_coordinates, img, margin):
    # 对旋转后的图像左右下按照landmarks裁剪，上部按margin保留部分额头
        left, right = np.min(rot_coordinates, axis=0)[0], np.max(rot_coordinates, axis=0)[0]
        top, bottom = np.min(rot_coordinates, axis=0)[1], np.max(rot_coordinates, axis=0)[1]
        kept = top - int((bottom - top) * margin)
        if kept < 0:
            kept = 0
        if left < 0:
            left = 0
        if bottom > img.shape[0]:
            bottom = img.shpe[0]
        if right > img.shape[1]:
            right = img.shape[1]

        crop_img = img[kept:bottom + 1, left:right + 1]
        return crop_img

if __name__ == '__main__':
    face_dector = preprocess_landmarks()
    face_rotate = preprocess_rotate()
    face_crop = preprocess_crop()
    img_path = "F:\datasets\MMEW\MMEW\Micro_Expression\surprise\S29-02-001/1.jpg"

    coordinates = face_dector.detect(img_path)  # 返回人脸关键点landmarks
    angle = face_rotate.eye_angle(coordinates)  # 返回人眼测得所需摆正角度
    img = cv2.imread(img_path)
    rot_img, rot_coordinates = face_rotate.rotate(img, angle, coordinates)  # 摆正图片及landmarks
    crop_img = face_crop.crop(rot_coordinates, img, 0.2)  # 裁剪人脸图片
    cv2.imshow("win", crop_img)
    cv2.waitKey(0)
    # print(coordinates[0])
    print(angle)
    print(rot_img.shape)
    print(rot_coordinates)
    print(crop_img.shape)
    cv2.imwrite("done.jpg", crop_img)
    cv2.imwrite("original.jpg", img)