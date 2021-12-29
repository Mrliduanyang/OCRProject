import argparse
import json
from pathlib import Path

from functools import cmp_to_key

import numpy as np
import cv2

license_template = {
    # name, sex, nation, birth, address, idNumber
    'id': [
        (70, 140, 150, 300),
        (150, 220, 150, 220),
        (150, 220, 330, 390),
        (220, 290, 150, 470),
        (290, 350, 150, 550),
        (350, 420, 150, 550),
        (480, 550, 280, 800),
    ],

    # idNum, name, sex, nationality, address, trafficOrganization, birthDate, firstIssueDate, class, validPeriodStart, validPeriodEnd
    'driver': [
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
    ],

    # plateNo, vehicleType, owner, address, useCharacter, model, trafficOrganization, vin, engineNo, registerDate, issueDate
    'vehicle': [
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
}


def perspective_transform(img, bounding_vertex, target_size):
    width, height = target_size
    bounding_vertex = np.array(bounding_vertex, dtype=np.float32)
    target_vertex = np.array([
        [0, 0],
        [0, height],
        [width, height],
        [width, 0],
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(bounding_vertex, target_vertex)
    ret = cv2.warpPerspective(img, M, target_size)
    return ret


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('license_type', help='license type')
    parser.add_argument('input_dir', help='input license image directory')
    parser.add_argument('output_dir', help='OCR image output directory')
    return parser.parse_args()


def crop_info_area(img, pos, save_path):
    y0, y1, x0, x1 = pos
    area = img[y0:y1, x0:x1]
    cv2.imwrite(f'{save_path}', area)

    return area


def sort_points(points):
    # 找到两个方向的中值
    points = np.array(points)
    mid_x = np.mean(points[:, 0])
    mid_y = np.mean([points[:, 1]])
    for point in points:
        x, y = point
        if x < mid_x and y < mid_y:
            lb = point
        elif x < mid_x and y > mid_y:
            lt = point
        elif x > mid_x and y > mid_y:
            rt = point
        elif x > mid_x and y < mid_y:
            rb = point
    return np.array([lb, lt, rt, rb], dtype=np.float32)


def main(args):
    license_type = args.license_type
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        Path.mkdir(output_dir)
        print('Creating OCR image output directory:', output_dir)

    for label_file in Path(args.input_dir).glob('*.json'):
        print('Processing:', label_file.stem)
        with open(label_file, 'r') as f:
            data = json.load(f)
            # 图像路径
            img_file = input_dir.joinpath(data['imagePath'])
            img = cv2.imread(str(img_file))
            # 获取标记框的四个顶点
            points = sort_points(data['shapes'][0]['points'])
            target_vertex = np.array([
                [0, 0],
                [0, 600],
                [880, 600],
                [880, 0],
            ], dtype=np.float32)
            # 透视变换
            M = cv2.getPerspectiveTransform(points, target_vertex)
            crop_img = cv2.warpPerspective(img, M, (880, 600))
            for i, pos in enumerate(license_template[license_type]):
                save_path = f'{img_file.stem}_{i}.jpg'
                crop_info_area(crop_img, pos, str(
                    output_dir.joinpath(save_path)))


if __name__ == '__main__':
    args = parse_args()
    main(args)
