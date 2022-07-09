import face_alignment
import cv2
from skimage import io

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda')

def enumerate_path():
    root = "/data/weimengting/datasets/MMEW/MMEW/Micro_Expression"
    test_img = "F:\datasets\MMEW\MMEW\Micro_Expression\disgust\S04-03-001/1.jpg"
    # 输出为(68, 2)的关键点矩阵
    input = io.imread(test_img)
    preds = fa.get_landmarks(input)
    img = cv2.imread(test_img)
    return img, preds

'''
    将计算得到的关键点在原图上面画出来
'''
def plot_keys(img_src, landmarks):
    for id, point in enumerate(landmarks[0]):
        # 68点坐标
        # 获取x,y坐标
        pos = (int(point[0]), int(point[1]))
        print(id, pos)
        # 利用cv2.circle给每个特征点画一个圈，共68个,第二个位置上面的坐标点只能是整数，不然会报错
        cv2.circle(img_src, pos, 2, color=(0, 255, 0))
        # 利用cv2.putText输出1-68
        font = cv2.FONT_HERSHEY_SIMPLEX
        # 打印点的索引
        cv2.putText(img_src, str(id + 1), pos, font, 0.2, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow("win1", img_src)
    cv2.waitKey(0)


if __name__ == '__main__':
    print("done!")
    img_src, landmarks = enumerate_path()
    plot_keys(img_src, landmarks)