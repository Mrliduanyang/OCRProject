import re

#打印身份证结构化信息
class IDStructure():
    #该字段对应的内容
    boxContext = {}
    #标志该字段是否被分割
    splitFlag = {}
    #集成固定字段的文本框坐标
    boxDict = {}

    def __init__(self, res):
        self.res = res

    #分割检索到的字段
    def split_context(self, str, dataContext):
        self.splitFlag[str] = False
        lenStr = len(str)
        if dataContext.split(str[lenStr - 1])[1] != '':
            self.splitFlag[str] = True
            self.boxContext[str] = dataContext.split(str[lenStr - 1])[1]
        else:
            self.boxContext[str] = ''

    #查找指定字段的内容
    def find_context(self, key, dataBox):
        if self.splitFlag[key] == True:
            return
        dataTextList = []
        for dataText in self.res[0]['data']:
            if key == dataText['text']:
                continue
            temp_box_position = dataText['text_box_position']
            if abs(temp_box_position[0][1] - dataBox[0][1]) < 28 and abs(temp_box_position[3][1] - dataBox[3][1]) < 28 and temp_box_position[0][0] - dataBox[1][0] > -1:
                dataTextList.append(dataText)
        minX = 999999
        for text in dataTextList:
            if text['text_box_position'][0][0] < minX:
                minX = text['text_box_position'][0][0]
                self.boxContext[key] = text['text']

    def generate_boxDict(self):
        part2complete = {'姓名': '姓名',
        '性别': '性别',
        '出生': '出生',
        '住': '住址',
        '址': '住址',
        '身份': '公民身份号码',
        }
        for data in self.res[0]['data']:
            for key,value in part2complete.items():
                if key in data['text']:
                    self.boxDict[value] = data['text_box_position']
                    self.split_context(value, data['text'])
                    break

    def generate_structure(self):
        self.generate_boxDict()
        for key,value in self.boxDict.items():
            self.find_context(key, value)
        for key,value in self.boxDict.items():
            print('%s：%s'%(key, self.boxContext[key]))


#打印驾驶证结构化信息
class DStructure():
    #该字段对应的内容
    boxContext = {}
    #标志该字段是否被分割
    splitFlag = {}
    #集成固定字段的文本框坐标
    boxDict = {}

    def __init__(self, res):
        self.res = res

    #分割检索到的字段
    def split_context(self, str, dataContext):
        self.splitFlag[str] = False
        lenStr = len(str)
        if dataContext.split(str[lenStr - 1])[1] != '':
            self.splitFlag[str] = True
            self.boxContext[str] = dataContext.split(str[lenStr - 1])[1]
        else:
            self.boxContext[str] = ''

    #查找指定字段的内容
    def find_context(self, key, dataBox):
        if self.splitFlag[key] == True:
            return
        dataTextList = []
        for dataText in self.res[0]['data']:
            if key == dataText['text']:
                continue
            temp_box_position = dataText['text_box_position']
            if abs(temp_box_position[0][1] - dataBox[0][1]) < 28 and abs(temp_box_position[3][1] - dataBox[3][1]) < 28 and temp_box_position[0][0] - dataBox[1][0] > -1:
                dataTextList.append(dataText)
        minX = 999999
        for text in dataTextList:
            if text['text_box_position'][0][0] < minX:
                minX = text['text_box_position'][0][0]
                self.boxContext[key] = text['text']

    def generate_boxDict(self):
        part2complete = {'证号': '证号',
        '姓名': '姓名',
        '国籍': '国籍',
        '住': '住址',
        '址': '住址',
        '出生': '出生日期',
        '初次': '初次领证日期',
        '准驾': '准驾车型',
        '有效': '有效期限',
        '至': '至',
        }
        for data in self.res[0]['data']:
            for key,value in part2complete.items():
                if key in data['text']:
                    self.boxDict[value] = data['text_box_position']
                    self.split_context(value, data['text'])
                    break

    def generate_structure(self):
        self.generate_boxDict()
        for key,value in self.boxDict.items():
            self.find_context(key, value)
        #调整格式并输出
        dateList = ['出生日期', '初次领证日期', '有效期限', '至']
        for dateType in dateList:
            if '.' in self.boxContext[dateType]:
                    self.boxContext[dateType] = self.boxContext[dateType].replace('.', '-')
            self.boxContext[dateType] = re.sub('[A-Za-z\u4e00-\u9fa5]', '', self.boxContext[dateType])
        for key,value in self.boxDict.items():
            print('%s：%s'%(key, self.boxContext[key]))


#打印行驶证结构化信息
class VStructure():
    #该字段对应的内容
    boxContext = {}
    #标志该字段是否被分割
    splitFlag = {}
    #集成固定字段的文本框坐标
    boxDict = {}

    def __init__(self, res):
        self.res = res

    #分割检索到的字段
    def split_context(self, str, dataContext):
        self.splitFlag[str] = False
        lenStr = len(str)
        if dataContext.split(str[lenStr - 1])[1] != '':
            self.splitFlag[str] = True
            self.boxContext[str] = dataContext.split(str[lenStr - 1])[1]
        else:
            self.boxContext[str] = ''

    #查找指定字段的内容
    def find_context(self, key, dataBox):
        if self.splitFlag[key] == True:
            return
        dataTextList = []
        for dataText in self.res[0]['data']:
            if key == dataText['text']:
                continue
            temp_box_position = dataText['text_box_position']
            if abs(temp_box_position[0][1] - dataBox[0][1]) < 28 and abs(temp_box_position[3][1] - dataBox[3][1]) < 28 and temp_box_position[0][0] - dataBox[1][0] > -1:
                dataTextList.append(dataText)
        minX = 999999
        for text in dataTextList:
            if text['text_box_position'][0][0] < minX:
                minX = text['text_box_position'][0][0]
                self.boxContext[key] = text['text']

    def generate_boxDict(self):
        part2complete = {'号牌': '号牌号码',
        '类型': '车辆类型',
        '所有': '所有人',
        '住': '住址',
        '址': '住址',
        '性质': '使用性质',
        '型号': '品牌型号',
        '代号': '车辆识别代号',
        '发动': '发动机号码',
        '注册': '注册日期',
        '发证': '发证日期'
        }
        for data in self.res[0]['data']:
            for key,value in part2complete.items():
                if key in data['text']:
                    self.boxDict[value] = data['text_box_position']
                    self.split_context(value, data['text'])
                    break

    def generate_structure(self):
        self.generate_boxDict()
        for key,value in self.boxDict.items():
            self.find_context(key, value)
        #调整格式并输出
        dateList = ['注册日期', '发证日期']
        for dateType in dateList:
            if '.' in self.boxContext[dateType]:
                    self.boxContext[dateType] = self.boxContext[dateType].replace('.', '-')
            self.boxContext[dateType] = re.sub('[A-Za-z\u4e00-\u9fa5]', '', self.boxContext[dateType])
        for key,value in self.boxDict.items():
            print('%s：%s'%(key, self.boxContext[key]))


#根据图片选择指定模板打印结构化信息
class PrintStructure():
    def __init__(self, picture):
        self.picture = picture

    def print_structure(self):
        # dLicense = DStructure(self.picture)
        #key:指定字段; value:调用函数
        card2fun = {'公民身份号码': IDStructure(self.picture).generate_structure, 
        '驾驶证': DStructure(self.picture).generate_structure, 
        '行驶证': VStructure(self.picture).generate_structure}
        #标志是否可以打印
        printFlag = False
        for data in self.picture[0]['data']:
            if printFlag == True:
                break
            for key in card2fun.keys():
                if key in data['text']:
                    card2fun.get(key, None)()
                    printFlag = True
                    break
        if printFlag == False:
            print('请上传正确的证件图片！')
