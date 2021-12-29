# -*- coding:utf-8 -*-
import datetime

#校验字符串是否全由汉字组成
def str_isall_chinese(str):
    for ch in str:
        if not u'\u4e00' <= ch <= u'\u9fa5':
            return False
    return True

#校验日期字段是否为“YYYY-MM-DD”格式
def date_is_valid(date_text):
    try:
        if datetime.datetime.strptime(date_text, '%Y-%m-%d'):
            return True
        else:
            return False
    except:
        return False

#校验车牌号是否合规
def plate_is_valid(plateStr):
    if len(plateStr) != 7:
        return '识别错误，车牌号应为固定七位'
    if not u'\u4e00' <= plateStr[0] <= u'\u9fa5':
        return '识别错误，车牌号第一位应为汉字'
    if not plateStr[1].isalpha():
        return '识别错误，车牌号第二位应为字母'
    if not plateStr[2:7].isalnum():
        return '识别错误，车牌号后五位应为字母或数字'
    return plateStr

#校验VIN码是否合规
def vin_is_valid(vinStr):
    if len(vinStr) != 17:
        return '识别错误，VIN码应为固定十七位'
    if not vinStr.isalnum():
        return '识别错误，VIN码应为字母或数字'
    return vinStr

#校验身份证号是否合规
def idNum_is_valid(idNum):
    if len(idNum) != 18:
        return '识别错误，证号应为固定十八位'
    if not idNum[0:17].isdigit():
        return '识别错误，证号前十七位应为数字'
    if not (idNum[17].isdigit() or idNum[17] == 'X'):
        return '识别错误，证号最后一位应为数字或字母X'
    return idNum


#校验驾驶证信息是否合规
def dCard_is_valid(dCardDict):
    dCardKeyList = ['idNum', 'name', 'sex', 'nationality', 'address', 'birthDate', 'firstIssueDate', 'class', 'validPeriodStart', 'validPeriodEnd', 'trafficOrganization']
    for key in dCardKeyList:
        if key not in dCardDict:
            dCardDict[key] = ''

    if dCardDict['idNum'] != '':
        dCardDict['idNum'] = idNum_is_valid(dCardDict['idNum'])
    if dCardDict['name'] != '' and not str_isall_chinese(dCardDict['name']):
        dCardDict['name'] = '识别错误，姓名应由汉字组成'
    if dCardDict['sex'] != '' and not dCardDict['sex'] in ['男', '女']:
        dCardDict['sex'] = '识别错误，性别应为\'男\'或\'女\''
    if dCardDict['birthDate'] != '' and not date_is_valid(dCardDict['birthDate']):
        dCardDict['birthDate'] = dCardDict['birthDate'] + '识别错误，日期格式应为YYYY-MM-DD'
    if dCardDict['firstIssueDate'] != '' and not date_is_valid(dCardDict['firstIssueDate']):
        dCardDict['firstIssueDate'] = dCardDict['firstIssueDate'] + '识别错误，日期格式应为YYYY-MM-DD'
    if dCardDict['class'] != '' and not dCardDict['class'].isalnum():
        dCardDict['class'] = '识别错误，准驾车型应为字母或数字'
    if dCardDict['validPeriodStart'] != '' and not date_is_valid(dCardDict['validPeriodStart']):
        dCardDict['validPeriodStart'] = '识别错误，日期格式应为YYYY-MM-DD'
    if dCardDict['validPeriodEnd'] != '' and not date_is_valid(dCardDict['validPeriodEnd']):
        dCardDict['validPeriodEnd'] = dCardDict['validPeriodEnd'] + '识别错误，日期格式应为YYYY-MM-DD'
    if dCardDict['trafficOrganization'] != '' and not str_isall_chinese(dCardDict['trafficOrganization']):
        dCardDict['trafficOrganization'] = '识别错误，交通机构应由汉字组成'

    new_dCardDict = {}
    for key in dCardKeyList:
        if dCardDict.get(key) == '':
            new_dCardDict[key] = '未识别到该字段'
        else:
            new_dCardDict[key] = dCardDict.get(key)
    return new_dCardDict

#校验行驶证信息是否合规
def vCard_is_valid(vCardDict):
    vCardKeyList = ['plateNum', 'vehicleType', 'owner', 'address', 'useCharacter', 'model', 'vin', 'engineNum', 'registerDate', 'issueDate', 'trafficOrganization']
    for key in vCardKeyList:
        if key not in vCardDict:
            vCardDict[key] = ''

    if vCardDict['plateNum'] != '':
        vCardDict['plateNum'] = plate_is_valid(vCardDict['plateNum'])
    if vCardDict['owner'] != '' and not str_isall_chinese(vCardDict['owner']):
        vCardDict['owner'] = '识别错误，所有人应由汉字组成'
    if vCardDict['useCharacter'] != '' and not vCardDict['useCharacter'] in ['营运', '非营运', '营转非']:
        vCardDict['useCharacter'] = '识别错误，使用性质不是\'营运\', \'非营运\', \'营转非\'其中之一'
    if vCardDict['vin'] != '':
        vCardDict['vin'] = vin_is_valid(vCardDict['vin'])
    if vCardDict['engineNum'] != '' and not vCardDict['engineNum'].isalnum():
        vCardDict['engineNum'] = '发动机号码应为字母或数字'
    if vCardDict['registerDate'] != '' and not date_is_valid(vCardDict['registerDate']):
        vCardDict['registerDate'] = '识别错误，日期格式应为YYYY-MM-DD'
    if vCardDict['issueDate'] != '' and not date_is_valid(vCardDict['issueDate']):
        vCardDict['issueDate'] = '识别错误，日期格式应为YYYY-MM-DD'
    if vCardDict['trafficOrganization'] != '' and not str_isall_chinese(vCardDict['trafficOrganization']):
        vCardDict['trafficOrganization'] = '识别错误，交通机构应由汉字组成'

    new_vCardDict = {}
    for key in vCardKeyList:
        if vCardDict.get(key) == '':
            new_vCardDict[key] = '未识别到该字段'
        else:
            new_vCardDict[key] = vCardDict.get(key)
    return new_vCardDict

#校验身份证信息是否合规
def idCard_is_valid(idCardDict):
    idCardKeyList = ['idNum', 'name', 'sex', 'ethnicity', 'birthDate', 'address']
    for key in idCardKeyList:
        if key not in idCardDict:
            idCardDict[key] = ''

    if idCardDict['idNum'] != '':
        idCardDict['idNum'] = idNum_is_valid(idCardDict['idNum'])
    if idCardDict['name'] != '' and not str_isall_chinese(idCardDict['name']):
        idCardDict['name'] = '识别错误，姓名应由汉字组成'
    if idCardDict['sex'] != '' and not idCardDict['sex'] in ['男', '女']:
        idCardDict['sex'] = '识别错误，性别应为\'男\'或\'女\''
    if idCardDict['birthDate'] != '' and not date_is_valid(idCardDict['birthDate']):
        idCardDict['birthDate'] = '识别错误，日期格式应为YYYY-MM-DD'

    new_idCardDict = {}
    for key in idCardKeyList:
        if idCardDict.get(key) == '':
            new_idCardDict[key] = '未识别到该字段'
        else:
            new_idCardDict[key] = idCardDict.get(key)
    return new_idCardDict