# %%
import os
import re
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
from matplotlib import pyplot as plt
import cv2

import paddle
from paddleseg import transforms as T
from paddleseg.core import infer
from paddleseg.utils import visualize, utils
from paddleseg.models import BiSeNetV2

from paddleocr import PaddleOCR


class OCR:
    def __init__(self, id_path, driver_path, vehicle_path, det_path, rec_path, cls_path, rec_dict_path):
        # 加载三种证件的实例分割模型
        self.id_seg_model = BiSeNetV2(num_classes=2, lambd=0.25,
                                      align_corners=False, pretrained=None)
        utils.load_entire_model(
            self.id_seg_model, id_path)
        self.id_seg_model.eval()

        self.driver_seg_model = BiSeNetV2(num_classes=2, lambd=0.25,
                                          align_corners=False, pretrained=None)
        utils.load_entire_model(
            self.driver_seg_model, driver_path)
        self.driver_seg_model.eval()

        self.vehicle_seg_model = BiSeNetV2(num_classes=2, lambd=0.25,
                                           align_corners=False, pretrained=None)
        utils.load_entire_model(
            self.vehicle_seg_model, vehicle_path)
        self.vehicle_seg_model.eval()

        # 图像变换
        self._transforms = T.Compose([
            T.Resize(target_size=(512, 512)),
            T.Normalize()
        ])

        # 实例分割后的伪色图的颜色映射
        self._color_map = visualize.get_color_map_list(256)

        # 加载OCR模型
        self.ocr_model = PaddleOCR(det_model_dir=det_path, rec_model_dir=rec_path,
                                   rec_char_dict_path=rec_dict_path, cls_model_dir=cls_path)
        # 三类证件的信息区域模板
        # 从上到下，从左到右顺序
        # name, sex, nation, birth, address, idNumber
        self.id_template = [
            (70, 140, 150, 600),
            (150, 220, 150, 220),
            (150, 220, 330, 600),
            (220, 290, 150, 470),
            (290, 360, 150, 550),
            (350, 420, 150, 550),
            (480, 560, 280, 800),
        ]

        # name, idNum, sex, nationality, address, trafficOrganization, birthDate, firstIssueDate, class, validPeriodStart, validPeriodEnd
        self.driver_template = [
            (100, 150, 330, 750),
            (140, 220, 100, 410),
            (140, 220, 465, 560),
            (140, 220, 660, 850),
            (200, 260, 110, 850),
            (260, 320, 40, 650),

            (310, 380, 30, 230),
            (380, 440, 30, 230),
            (450, 510, 30, 230),

            (310, 390, 340, 620),
            (370, 440, 390, 650),
            (440, 510, 340, 640),
            (500, 590, 150, 390),
            (500, 590, 430, 660)
        ]

        # plateNum, vehicleType, owner, address, useCharacter, model, vin, engineNum, registerDate, issueDate
        self.vehicle_template = [
            (100, 180, 140, 380),
            (100, 180, 490, 850),
            (160, 250, 140, 850),
            (230, 320, 140, 850),
            (300, 390, 140, 320),
            (300, 390, 430, 870),

            (380, 440, 40, 240),
            (450, 510, 40, 240),
            (510, 570, 40, 240),

            (370, 470, 390, 870),
            (440, 530, 370, 870),
            (510, 590, 340, 550),
            (510, 590, 640, 870),
        ]

        # 车牌正则校验
        self.plate_pattern = re.compile(
            r'^([京津晋冀蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新][ABCDEFGHJKLMNPQRSTUVWXY][1-9DF][1-9ABCDEFGHJKLMNPQRSTUVWXYZ]\d{3}[1-9DF]|[京津晋冀蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新][ABCDEFGHJKLMNPQRSTUVWXY][\dABCDEFGHJKLNMxPQRSTUVWXYZ]{5})$')

        # VIN码正则校验
        self.vin_pattern = re.compile(r'^[A-HJ-NPR-Z\d]{17}$')

    def show_img(self, img, img_name):
        img_copy = np.copy(img)
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        cv2.imwrite(img_name, img)
        plt.imshow(img_copy)
        plt.show()

    # 获取伪色图
    def _get_pseudo_color_map(self, result):
        color_map = [self._color_map[i:i + 3]
                     for i in range(0, len(self._color_map), 3)]
        color_map = np.array(color_map).astype(np.uint8)
        # Use OpenCV LUT for color mapping
        c1 = cv2.LUT(result, color_map[:, 0])
        c2 = cv2.LUT(result, color_map[:, 1])
        c3 = cv2.LUT(result, color_map[:, 2])
        pseudo_img = np.dstack((c1, c2, c3))
        return pseudo_img

    # 获取实例分割掩码
    def _get_mask(self, model_name, img):
        with paddle.no_grad():
            ori_shape = img.shape[:2]
            img, _ = self._transforms(img)
            img = img[np.newaxis, ...]
            img = paddle.to_tensor(img)

            model = self.__dict__[model_name]

            pred = infer.inference(
                model,
                img,
                ori_shape=ori_shape,
                transforms=self._transforms.transforms)

            pred = paddle.squeeze(pred)
            pred = pred.numpy().astype(np.uint8)
            mask = self._get_pseudo_color_map(pred)
            return mask

    # 获得伪色图中证件部分四边形边界框的四个顶点
    def _get_bounding_vertex(self, pred_mask):
        # Canny算子检测边缘
        gray = cv2.cvtColor(pred_mask, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 75, 200)
        contours, hierarchy = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        if len(contours) == 0:
            raise Exception("无法检测到证件！")

        contour = contours[0]
        contour = np.squeeze(contour)
        x_mid = np.mean(contour[:, 0])
        y_mid = np.mean(contour[:, 1])

        # 上边缘部分，y < y_mid，中间4/5的点
        up_points = contour[contour[:, 1] < y_mid]
        x_min, x_max = np.min(up_points[:, 0]), np.max(up_points[:, 0])
        x_start, x_end = x_min + (x_max - x_min) / \
            10, x_max - (x_max - x_min) / 10
        up_edge = up_points[(up_points[:, 0] > x_start)
                            & (up_points[:, 0] < x_end)]
        # 根据点拟合直线，该直线可视为上边缘
        linear_model = np.polyfit(up_edge[:, 0], up_edge[:, 1], 1)
        up_line = np.poly1d(linear_model)

        # 右边缘部分，x > x_mid，选择中间4/5的点
        right_points = contour[contour[:, 0] > x_mid]
        y_min, y_max = np.min(right_points[:, 1]), np.max(right_points[:, 1])
        y_start, y_end = y_min + (y_max - y_min) / \
            10, y_max - (y_max - y_min) / 10
        right_edge = right_points[(right_points[:, 1] > y_start)
                                  & (right_points[:, 1] < y_end)]
        linear_model = np.polyfit(right_edge[:, 0], right_edge[:, 1], 1)
        right_line = np.poly1d(linear_model)

        # 下边缘部分，y > y_mid，选择中间4/5的点
        down_points = contour[contour[:, 1] > y_mid]
        x_min, x_max = np.min(down_points[:, 0]), np.max(down_points[:, 0])
        x_start, x_end = x_min + (x_max - x_min) / \
            10, x_max - (x_max - x_min) / 10
        down_edge = down_points[(down_points[:, 0] > x_start)
                                & (down_points[:, 0] < x_end)]
        linear_model = np.polyfit(down_edge[:, 0], down_edge[:, 1], 1)
        down_line = np.poly1d(linear_model)

        # 左边缘部分，x < x_mid，选择中间4/5的点
        left_points = contour[contour[:, 0] < x_mid]
        y_min, y_max = np.min(left_points[:, 1]), np.max(left_points[:, 1])
        y_start, y_end = y_min + (y_max - y_min) / \
            10, y_max - (y_max - y_min) / 10
        left_edge = left_points[(left_points[:, 1] > y_start)
                                & (left_points[:, 1] < y_end)]
        linear_model = np.polyfit(left_edge[:, 0], left_edge[:, 1], 1)
        left_line = np.poly1d(linear_model)

        # 获得上、右、下、左四条边缘拟合直线的交点，可视为证件的四个顶点
        left_up_x = np.roots(up_line - left_line)[0]
        left_down_x = np.roots(down_line - left_line)[0]
        right_down_x = np.roots(down_line - right_line)[0]
        right_up_x = np.roots(up_line - right_line)[0]

        left_up_y = up_line(left_up_x)
        left_down_y = down_line(left_down_x)
        right_down_y = down_line(right_down_x)
        right_up_y = up_line(right_up_x)

        # 四个顶点的坐标
        return np.array([[left_up_x, left_up_y],
                         [right_up_x, right_up_y],
                         [right_down_x, right_down_y],
                         [left_down_x, left_down_y],
                         ], dtype=np.float32)

    # 根据四角顶点做透视变换
    def _perspective_transform(self, img, bounding_vertex, target_size):
        width, height = target_size
        target_vertex = np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height],
        ], dtype=np.float32)
        M = cv2.getPerspectiveTransform(bounding_vertex, target_vertex)
        ret = cv2.warpPerspective(img, M, target_size)
        return ret

    def _recognize_info_areas(self, template_name, img):
        template = self.__dict__[template_name]

        def crop_info_area(pos):
            y0, y1, x0, x1 = pos
            area = img[y0:y1, x0:x1]
            return area

        ret = []
        for pos in template:
            info_area = crop_info_area(pos)
            recog_res = self.ocr_model.ocr(info_area)
            if len(recog_res) == 0:
                ret.append("")
            else:
                ret.append(recog_res[0][1][0])

        return ret

    def _recognize(self, model_name, template_name, img):
        # 获取证件的四边形边界框
        mask = self._get_mask(model_name, img)
        bounding_vertex = self._get_bounding_vertex(mask)
        # 透视变换
        crop_img = self._perspective_transform(
            img, bounding_vertex, (880, 600))
        template = self.__dict__[template_name]
        for info_area in template:
            cv2.rectangle(
                crop_img, (info_area[2], info_area[0]), (info_area[3], info_area[1]), (255, 0, 0), 2)
        self.show_img(crop_img, 'crop.png')
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        # 获取证件信息
        ret = self._recognize_info_areas(template_name, crop_img)
        return ret

    def recognize_id(self, img):
        recog_res = self._recognize('id_seg_model', 'id_template', img)
        ret = {
            'name': recog_res[0],
            'sex': recog_res[1],
            'ethnicity': recog_res[2],
            'birthDate': recog_res[3],
            'address': f'{recog_res[4]}{recog_res[5]}',
            'idNum': recog_res[6]
        }
        return ret

    def recognize_driver(self, img):
        recog_res = self._recognize('driver_seg_model', 'driver_template', img)
        ret = {
            'idNum': recog_res[0],
            'name': recog_res[1],
            'sex': recog_res[2],
            'nationality': recog_res[3],
            'address': f'{recog_res[4]}{recog_res[5]}',
            'trafficOrganization': f'{recog_res[6]}{recog_res[7]}{recog_res[8]}',
            'birthDate': recog_res[9],
            'firstIssueDate': recog_res[10],
            'class': recog_res[11],
            'validPeriodStart': recog_res[12],
            'validPeriodEnd': recog_res[13],
        }
        return ret

    def recognize_vehicle(self, img):
        recog_res = self._recognize(
            'vehicle_seg_model', 'vehicle_template', img)
        # plateNo, vehicleType, owner, address, useCharacter, model, trafficOrganization, vin, engineNo, registerDate, issueDate
        ret = {
            'plateNum': recog_res[0],
            'vehicleType': recog_res[1],
            'owner': recog_res[2],
            'address': recog_res[3],
            'useCharacter': recog_res[4],
            'model': recog_res[5],
            'trafficOrganization': f'{recog_res[6]}{recog_res[7]}{recog_res[8]}',
            'vin': recog_res[9],
            'engineNum': recog_res[10],
            'registerDate': recog_res[11],
            'issueDate': recog_res[12],
        }
        return ret

    def recognize_plate(self, img):
        recog_res = self.ocr_model.ocr(img)
        for res in recog_res:
            if self.plate_pattern.match(res[1][0]):
                return res[1][0]
        raise Exception("未识别到车牌！")

    def recognize_vin(self, img):
        recog_res = self.ocr_model.ocr(img)
        for res in recog_res:
            if self.vin_pattern.match(res[1][0]):
                return res[1][0]
        raise Exception("未识别到VIN码！")
