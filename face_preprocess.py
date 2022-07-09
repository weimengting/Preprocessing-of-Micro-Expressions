import numpy as np
import cv2
import dlib
from math import *

class preprocess_landmarks(object):

    def landmark_to_np(self, landmarks, dtype='int'):
    # 将dlib格式的关键点转换成numpy格式
        coordinates = np.zeros((landmarks.num_parts, 2), dtype=dtype) # 创建68*2用于存放坐标
        for i in range(0, landmarks.num_parts):
            coordinates[i] = (landmarks.part(i).x, landmarks.part(i).y)

        return coordinates

    def detect(self, img):
    # 返回68个人脸关键点坐标
        detector = dlib.get_frontal_face_detector() # 人脸搜索器
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') # 关键点检测器
        rects = detector(img, 0)  # 返回人脸检测框[左右上下]
        # for index, edges in enumerate(rects):  # 遍历一张图的每个人脸
        landmarks = predictor(img, rects[0]) # 此处应使用for循环遍历图中的每张人脸，但作为MER任务只处理单张人脸
        coordinates = self.landmark_to_np(landmarks)

        return coordinates

class preprocess_rotate(object):

    def eye_angle(self, coordinates):
    # 利用左右眼各6个landmarks求取图片摆正所需旋转角度
        eyel_x = np.round(
            np.mean([coordinates[36, 0], coordinates[37, 0], coordinates[38, 0], coordinates[39, 0], coordinates[40, 0],
                     coordinates[41, 0]])).astype(np.int)
        eyel_y = np.round(
            np.mean([coordinates[36, 1], coordinates[37, 1], coordinates[38, 1], coordinates[39, 1], coordinates[40, 1],
                     coordinates[41, 1]])).astype(np.int)
        eyer_x = np.round(
            np.mean([coordinates[42, 0], coordinates[43, 0], coordinates[44, 0], coordinates[45, 0], coordinates[46, 0],
                     coordinates[47, 0]])).astype(np.int)
        eyer_y = np.round(
            np.mean([coordinates[42, 1], coordinates[43, 1], coordinates[44, 1], coordinates[45, 1], coordinates[46, 1],
                     coordinates[47, 1]])).astype(np.int)
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
        crop_img = img[top - int((bottom - top) * margin):bottom + 1, left:right + 1]

        return crop_img