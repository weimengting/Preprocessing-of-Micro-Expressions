from crop_face_by_face_align import *
import os

'''
    遍历路径，预处理微表情
'''
class MicroExpression():
    def __init__(self, margin):
        self.margin = margin
        self.face_dector = preprocess_landmarks()
        self.face_rotate = preprocess_rotate()
        self.face_crop = preprocess_crop()

    def glob_path(self, root):
        categories = os.listdir(root)
        for category in categories:
            cur_category_path = os.path.join(root, category)
            subjects = os.listdir(cur_category_path)
            for subject in subjects:
                cur_subject_path = os.path.join(cur_category_path, subject)
                imgs = os.listdir(cur_subject_path)
                tmp_list = []
                for img in imgs:
                    tmp_name = int(img.replace('.jpg', ''))
                    tmp_list.append(tmp_name)
                anchor_name = str(tmp_list[0]) + '.jpg'
                anchor_img_path = os.path.join(cur_subject_path, anchor_name)

                anchor_img = cv2.imread(anchor_img_path)
                coordinates = self.face_dector.detect(anchor_img_path)  # 返回人脸关键点landmarks
                angle = self.face_rotate.eye_angle(coordinates)  # 返回人眼测得所需摆正角度

                rot_img, rot_coordinates = self.face_rotate.rotate(anchor_img, angle, coordinates)  # 摆正图片及landmarks
                crop_img = self.face_crop.crop(rot_coordinates, anchor_img, self.margin)  # 裁剪人脸图片
                save_root = 'D:\pycharm_projects\pythonProject\local\preprocess_MMEW'
                save_path = os.path.join(save_root, 'MicroExpression', category, subject)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                cv2.imwrite(os.path.join(save_path, anchor_name), crop_img)
                for i in range(1, len(tmp_list)):
                    cur_img_name = str(tmp_list[i]) + '.jpg'
                    cur_img_path = os.path.join(cur_subject_path, cur_img_name)
                    cur_img = cv2.imread(cur_img_path)
                    crop_cur_img = self.face_crop.crop(rot_coordinates, cur_img, self.margin)
                    cv2.imwrite(os.path.join(save_path, cur_img_name), crop_cur_img)
                    print("done!")


class MacroExpression():
    def __init__(self, margin):
        self.margin = margin
        self.face_dector = preprocess_landmarks()
        self.face_rotate = preprocess_rotate()
        self.face_crop = preprocess_crop()

    def glob_path(self, root):
        subjects = os.listdir(root)
        for subject in subjects:
            cur_subject_path = os.path.join(root, subject)
            categories = os.listdir(cur_subject_path)
            for category in categories:
                cur_category_path = os.path.join(cur_subject_path, category)
                imgs = sorted(os.listdir(cur_category_path))

                anchor_name = imgs[0]
                anchor_img_path = os.path.join(cur_category_path, anchor_name)

                anchor_img = cv2.imread(anchor_img_path)
                coordinates = self.face_dector.detect(anchor_img_path)  # 返回人脸关键点landmarks
                angle = self.face_rotate.eye_angle(coordinates)  # 返回人眼测得所需摆正角度

                rot_img, rot_coordinates = self.face_rotate.rotate(anchor_img, angle, coordinates)  # 摆正图片及landmarks
                crop_img = self.face_crop.crop(rot_coordinates, anchor_img, self.margin)  # 裁剪人脸图片
                save_root = 'D:\pycharm_projects\pythonProject\local\preprocess_MMEW'
                save_path = os.path.join(save_root, 'MacroExpression', subject, category)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                cv2.imwrite(os.path.join(save_path, anchor_name), crop_img)
                for i in range(1, len(imgs)):
                    cur_img_name = imgs[i]
                    cur_img_path = os.path.join(cur_category_path, cur_img_name)
                    cur_img = cv2.imread(cur_img_path)
                    crop_cur_img = self.face_crop.crop(rot_coordinates, cur_img, self.margin)
                    cv2.imwrite(os.path.join(save_path, cur_img_name), crop_cur_img)
                    print("done!")

class LiuTest():
    def __init__(self, margin):
        self.margin = margin
        self.face_dector = preprocess_landmarks()
        self.face_rotate = preprocess_rotate()
        self.face_crop = preprocess_crop()

    def glob_path(self, root):
        subjects = os.listdir(root)
        for subject in subjects:
            cur_subject_path = os.path.join(root, subject)
            imgs = sorted(os.listdir(cur_subject_path))

            anchor_name = imgs[0]
            print(anchor_name)
            anchor_img_path = os.path.join(cur_subject_path, anchor_name)

            anchor_img = cv2.imread(anchor_img_path)
            coordinates = self.face_dector.detect(anchor_img_path)  # 返回人脸关键点landmarks
            angle = self.face_rotate.eye_angle(coordinates)  # 返回人眼测得所需摆正角度

            rot_img, rot_coordinates = self.face_rotate.rotate(anchor_img, angle, coordinates)  # 摆正图片及landmarks
            crop_img = self.face_crop.crop(rot_coordinates, anchor_img, self.margin)  # 裁剪人脸图片
            save_root = 'E://wei_cropped//SAMM'
            save_path = os.path.join(save_root, subject)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imwrite(os.path.join(save_path, anchor_name), crop_img)
            for i in range(1, len(imgs)):
                cur_img_name = imgs[i]
                cur_img_path = os.path.join(cur_subject_path, cur_img_name)
                cur_img = cv2.imread(cur_img_path)
                crop_cur_img = self.face_crop.crop(rot_coordinates, cur_img, self.margin)
                cv2.imwrite(os.path.join(save_path, cur_img_name), crop_cur_img)
                print("done!")



if __name__ == '__main__':
    ME = LiuTest(margin=0.2)
    ME.glob_path(root='E://SAMM_Test_cropped')