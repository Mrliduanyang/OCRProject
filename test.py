# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2
from ocr import OCR

ocr = OCR(id_path='./models/id_seg_infer.pdparams', driver_path='./models/driver_seg_infer.pdparams',
          vehicle_path='./models/vehicle_seg_infer.pdparams',
          det_path='./models/ch_ppocr_server_v2.0_det_infer',
          rec_path='./models/ch_ppocr_server_v2.0_rec_infer',
          cls_path='./models/ch_ppocr_mobile_v2.0_cls_infer',
          rec_dict_path='./models/ppocr_keys_v1.txt')


# 身份证规格 8.56cm * 5.4cm
# 驾驶证规格 8.8cm * 6cm
# 行驶证规格 8.8cm * 6cm

# %%
print(ocr.recognize_id(cv2.imread(
'data/license_seg/id/imgs/161-8E4D9037-5593-ADA4-62A4-8045C51DD13D.jpg')))
print(ocr.recognize_driver(cv2.imread(
    'data/license_seg/driver/imgs/161-205BFD72-2A91-335B-F8E8-2F1A4020D8EC.jpg')))
print(ocr.recognize_vehicle(cv2.imread(
    'data/license_seg/vehicle/imgs/161-76E8E905-7947-88CC-4624-F92D06F9D4A6.jpg')))
# print(ocr.recognize_vin(cv2.imread('data/test/vin.jpg')))
# print(ocr.recognize_plate(cv2.imread('data/test/plate.jpg')))

# %%
