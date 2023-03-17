"""
    The main class for processing a database, i.e. MMEW, by enumerating the root directory.
    You can rewrite this according to your requirement since the file path and location mode of samples are various.
    Note that for the same video clip, we only compute the landmarks for the first frame, the remaining frames are aligned and rotated by
    the landmarks of the first frame. This is because the position of the camera rarely changes in the same video clip.

    Author : Mengting Wei
"""

from crop_face_by_face_align import *
import os


class MicroExpression():
    def __init__(self, margin):
        '''
            :param margin: ratio for keeping some facial area
        '''
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
                coordinates = self.face_dector.detect(anchor_img_path)
                angle = self.face_rotate.eye_angle(coordinates)

                rot_img, rot_coordinates = self.face_rotate.rotate(anchor_img, angle, coordinates)
                crop_img = self.face_crop.crop(rot_coordinates, anchor_img, self.margin)
                save_root = '....\preprocess_MMEW'
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
                coordinates = self.face_dector.detect(anchor_img_path)
                angle = self.face_rotate.eye_angle(coordinates)

                rot_img, rot_coordinates = self.face_rotate.rotate(anchor_img, angle, coordinates)
                crop_img = self.face_crop.crop(rot_coordinates, rot_img, self.margin)
                save_root = '....\preprocess_MMEW'
                save_path = os.path.join(save_root, 'MacroExpression', subject, category)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                cv2.imwrite(os.path.join(save_path, anchor_name), crop_img)
                for i in range(1, len(imgs)):
                    cur_img_name = imgs[i]
                    cur_img_path = os.path.join(cur_category_path, cur_img_name)
                    cur_img = cv2.imread(cur_img_path)
                    cur_rot_img, cur_rot_coordinates = self.face_rotate.rotate(cur_img, angle, coordinates)
                    crop_cur_img = self.face_crop.crop(cur_rot_coordinates, cur_rot_img, self.margin)
                    cv2.imwrite(os.path.join(save_path, cur_img_name), crop_cur_img)
                    print("done!")





if __name__ == '__main__':
    ME = MicroExpression(margin=0.2)
    ME.glob_path(root='..')