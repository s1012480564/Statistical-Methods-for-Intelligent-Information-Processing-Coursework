import numpy as np
import cv2
import glob
import logging
import sys

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def calibrate(f_names: list[str]) -> None:
    objp = np.zeros((9 * 9, 3), np.float32)
    x, y = np.meshgrid(np.arange(0, 9), np.arange(0, 9))
    objp[:, 0], objp[:, 1] = x.flatten(), y.flatten()
    obj_points = [objp] * len(f_names)
    img_points = []

    for f_name in f_names:
        img = cv2.imread(f_name)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, corners = cv2.findChessboardCorners(img_gray, (9, 9), None)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(img_gray, corners, (5, 5), (-1, -1), criteria)
        cv2.drawChessboardCorners(img, (9, 9), corners, True)
        cv2.imwrite("corner_images/" + f_name[f_name.rfind('\\') + 1:], img)

        img_points.append(corners2)

    h, w = cv2.imread(f_names[0]).shape[:2]

    cv2.destroyAllWindows()

    ret, mtx, dist, r_vecs, t_vecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)

    logger.info("ret: {}\nmtx: {}\ndist: {}\nr_vecs: {}\nt_vecs: {}\n".format(ret, mtx, dist, r_vecs, t_vecs))

    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    x, y, w, h = roi
    logger.info("new_camera_mtx: {}\nroi: {}".format(new_camera_mtx, roi))

    for f_name in f_names:
        img = cv2.imread(f_name)
        dst = cv2.undistort(img, mtx, dist, None, new_camera_mtx)
        dst1 = dst[y:y + h, x:x + w]
        cv2.imwrite("result_" + f_name, dst1)


if __name__ == '__main__':

    for i in [1, 2, 4, 5]:
        file_handler = logging.FileHandler("results_cam" + str(i) + ".log")
        logger.addHandler(file_handler)
        calibrate(glob.glob(r"images/images_cam" + str(i) + "/*.jpg"))
        logger.removeHandler(file_handler)
