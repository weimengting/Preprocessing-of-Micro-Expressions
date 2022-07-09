"""
    A test for computing and plotting facial landmarks.

"""
import face_alignment
import cv2
from skimage import io

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda')

def enumerate_path():
    test_img = "..../1.jpg"
    # output the landmarks matrix with shape=(68, 2)
    input = io.imread(test_img)
    preds = fa.get_landmarks(input)
    img = cv2.imread(test_img)
    return img, preds


def plot_landmarks(img_src, landmarks):
    # plot the landmarks on the image
    for id, point in enumerate(landmarks[0]):
        pos = (int(point[0]), int(point[1]))
        # circle each landmark
        cv2.circle(img_src, pos, 2, color=(0, 255, 0))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_src, str(id + 1), pos, font, 0.2, (0, 0, 0), 1, cv2.LINE_AA)
    # show the image
    cv2.imshow("win1", img_src)
    cv2.waitKey(0)


if __name__ == '__main__':
    img_src, landmarks = enumerate_path()
    plot_landmarks(img_src, landmarks)