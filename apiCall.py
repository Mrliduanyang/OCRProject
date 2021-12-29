# -*- coding:utf-8 -*-
from fastapi import FastAPI, File, UploadFile
import numpy
import uvicorn
import codecs
import isValid
from ocr import OCR
import cv2

import os
import re
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

app = FastAPI(title="中汽研OCR识别测试文档",
              #   description="",
              docs_url="/my_docs",
              openapi_url="/my_openapi")

ocr = OCR(id_path='./models/id_seg_infer.pdparams', driver_path='./models/driver_seg_infer.pdparams',
          vehicle_path='./models/vehicle_seg_infer.pdparams',
          det_path='./models/ch_ppocr_server_v2.0_det_infer',
          rec_path='./models/ch_ppocr_server_v2.0_rec_infer',
          cls_path='./models/ch_ppocr_server_v2.0_cls_infer',
          rec_dict_path='./models/ppocr_keys_v1.txt')

# 车牌号识别
@app.post("/plateRecog")
async def plate(file: bytes = File(...)):
    plateStr = ocr.recognize_plate(cv2.imdecode(numpy.frombuffer(file, dtype=numpy.uint8), cv2.IMREAD_COLOR))
    return isValid.plate_is_valid(plateStr)

# VIN码识别
@app.post("/vinRecog")
async def vin(file: bytes = File(...)):
    vinStr = ocr.recognize_vin(cv2.imdecode(numpy.frombuffer(file, dtype=numpy.uint8), cv2.IMREAD_COLOR))
    return isValid.vin_is_valid(vinStr)

# 驾驶证识别
@app.post("/dCardRecog")
async def dCard(file: bytes = File(...)):
    dCardDict = ocr.recognize_driver(cv2.imdecode(numpy.frombuffer(file, dtype=numpy.uint8), cv2.IMREAD_COLOR))
    return isValid.dCard_is_valid(dCardDict)

# 行驶证识别
@app.post("/vCardRecog")
async def vCard(file: bytes = File(...)):
    vCardDict = ocr.recognize_vehicle(cv2.imdecode(numpy.frombuffer(file, dtype=numpy.uint8), cv2.IMREAD_COLOR))
    return isValid.vCard_is_valid(vCardDict)

# 身份证识别
@app.post("/idCardRecog")
async def idCard(file: bytes = File(...)):
    idCardDict = ocr.recognize_id(cv2.imdecode(numpy.frombuffer(file, dtype=numpy.uint8), cv2.IMREAD_COLOR))
    return isValid.idCard_is_valid(idCardDict)

if __name__ == "__main__":
    uvicorn.run("apiCall:app", host="0.0.0.0", port=8080)
