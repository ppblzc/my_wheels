##################################### 功能描述 ###########################################
#
# 对内胶原图所属的分段图类型进行固定点切割，resize
# 获取图像变换后的对应xml
#
##########################################################################################


import cv2, os, glob, shutil, fire, random, time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scipy import stats,signal
from scipy.signal import argrelmin, argrelmax
from scipy.spatial import distance as dist

import xml.dom.minidom as minidom
try:
    from xml.etree import cElementTree as et
except:
    from xml.etree import ElementTree as et


"""
zS0ng
"""
############################ Data_parameters ####################################
Data_parameters = {
    "data_dir": r"D:\imgdata\zj\neijiao\neijiao_trainData\all_raw_trainData",
    "save_dir": "resized_1800",              
    "epochs":1,                 # 生成数据倍数，min>=1
    #######缺陷类别##########
    "defect_names":{
        "xuhan":0,
        "xipanyin":1,
        "huahen":2,
        "juanjiao":3,
        "quejiao":4,
        "other":5,
        "maoci":6

    },
    #######输入大小#########
    "input_resize":{
        "state":False,
        "input_resize_size":[1200,1800], #col,row
    },

    #######多区域分类剪切#########
    "corp":{
        "state":True,
        "region":{
            "A":{
                "row_lines":[1300,1500],    # h
                "col_lines":[450,3840],     # w
            },
            "B":{
                "row_lines":[1250,1450],    # h
                "col_lines":[450,3840],       # w
            },
            "C":{
                "row_lines":[1300,1500],    # h
                "col_lines":[0,3340],     # w
            },
            "D":{
                "row_lines":[1250,1450],    # h
                "col_lines":[0,3340],     # w
            },
            "E":{
                "row_lines":[1550,1750],    # h
                "col_lines":[50,5300],     # w
            },
            "F":{
                "row_lines":[1750,1950],    # h
                "col_lines":[200,5400],     # w
            },

        },

        "random":{              
            "state":False,      #在切线的基础上，配置左右浮动参数，建议不要太大。
            "row_random":10,
            "col_random":10,
        }
    },
    ###### 切分段 ######
    "cut_section":{
        "state":False,
        "col_section":350,   # 段宽度

    },
    #######翻转#########
    "flip":{
        "state":False,
        "flip":1,               # 1 水平翻转 0 垂直翻转 -1 水平垂直翻转 
        "random":{              
            "state":False,      # 随机水平垂直翻转
        }
    },
    #######旋转#########
    "rotate":{
        "state":False,
        "rotation_range": 170    # 以图片中心为旋转中心，旋转任意角度
    },
    #######仿射变换#########
    "warp_aﬃne":{               # 仿射变换
        "state":False,
        "pts1": [[10,10],[490,490],[490,0]],
        "pts2": [[0,0],[500,500],[500,0]],
    },
    #######透视变换#########
    "warpPerspective":{         # 透视变换
        "state":False,
        "pts1": [[10,10],[490,490],[490,0],[0,490]],
        "pts2": [[0,0],[500,500],[500,0],[0,500]],
    },
    #######输出大小#########
    "output_resize":{
        "state":True,
        "output_resize_size":[1800,200], #col,row
    },
    # 像素变换
    # "mean_rgb": [127.5, 127.5, 127.5],
    # "image_distort_strategy": {
    #     "expand_prob": 0.5,
    #     "expand_max_ratio": 4,
    #     "hue_prob": 0.5,
    #     "hue_delta": 18,
    #     "contrast_prob": 0.5,
    #     "contrast_delta": 0.5,
    #     "saturation_prob": 0.5,
    #     "saturation_delta": 0.5,
    #     "brightness_prob": 0.5,
    #     "brightness_delta": 0.125
    # },

}


def xml_to_pic(parsexml):
    root = parsexml.getroot()
    width = int(root.find('size').find('width').text)
    height = int(root.find('size').find('height').text)
    #############################
    num = len(root.findall('object'))
    img = np.zeros((height,width,num))
    img[img==0]=255

    defect_names=Data_parameters['defect_names']

    for i,obj in enumerate(root.findall('object')):
        bndbox = obj.find('bndbox')
        name = obj.find('name').text
        col_min = int(bndbox.find('xmin').text)
        col_max = int(bndbox.find('xmax').text)
        row_min = int(bndbox.find('ymin').text)
        row_max = int(bndbox.find('ymax').text)
        img[row_min:row_max,col_min:col_max,i] = defect_names[name]
    
    return img


def pic_to_xml(parsexml,img,_name):
    defect_names=Data_parameters['defect_names'] 
    root = parsexml.getroot()
    root.find('filename').text = _name+".jpg"
    root.find('path').text = _name+".jpg"
    root.find('size').find('width').text = str(img.shape[1])
    root.find('size').find('height').text = str(img.shape[0])
    
    for i,obj in enumerate(root.findall('object')):
        try:
            _img = img[:,:,i]
            # plt.imshow(img_, cmap=plt.cm.gray)
            # plt.show()
            img_index = np.where(_img != 255) #找到区域
            index_matrix = np.dstack((img_index[0], img_index[1])).squeeze()
            ############坐标###############
            _name = np.min(_img)
            name = list(defect_names.keys())[list(defect_names.values()).index(_name)]
            ############ xml ###############
            # print("pic_to_xml:::",index_matrix[0],index_matrix[-1])
            bndbox = obj.find('bndbox')
            obj.find('name').text = str(name)
            bndbox.find('xmin').text = str(index_matrix[0][1])
            bndbox.find('ymin').text = str(index_matrix[0][0])
            bndbox.find('xmax').text = str(index_matrix[-1][1])
            bndbox.find('ymax').text = str(index_matrix[-1][0])
        #######################针对不在切割范围内的缺陷################################
        except:
            root.remove(obj)
    return parsexml

# 判断分段图类型ABCD
def region_class(img_name):
    img_idx = os.path.basename(img_name).split('_')[-1]
    print('img_idx', img_idx)
    if img_idx == '0' or img_idx == '2':
        img_mark = 'A'
    elif img_idx == '5' or img_idx == '7':
        img_mark = 'B'
    elif img_idx == '1' or img_idx == '3':
        img_mark = 'C'
    elif img_idx == '4' or img_idx == '6':
        img_mark = 'D'
    elif img_idx == '8':
        img_mark = 'E'
    else:
        img_mark = 'F'
    return img_mark

# 切分段
def cut_pics(img_name, **kw):
    pass 

class Preprocess:
    def __init__(self):
        self.row_lines = []
        self.col_lines = []     

    def preprocess(self,img,mode,region):
        """
        数据处理包含三大类：
        - 像素变化：亮度，饱和度，反相，减淡、锐化、Gamma补偿等
        - 位置变换：水平翻转，垂直翻转，垂直水平翻转，random_crop（随机切微小值）
        - 切图：通常为固定方式，根据col_lins,row_lines切图

        数据校验包含以下方面：
        - 输入文件的名字格式
        - 输入图片的尺寸，通道数，mode（RGB），jpg
        - 输出的XML
        """
        ##############################################################
        ##############################################################
        nums = img.shape[2]
        print("nums:::",nums)

        if Data_parameters['input_resize']["state"] == True:
            resize_size = Data_parameters['input_resize']["input_resize_size"]
            img = cv2.resize(img, (resize_size[0], resize_size[1]),interpolation=cv2.INTER_NEAREST)
            img = np.reshape(img,(resize_size[1], resize_size[0],nums))
            print("input_resize:::",img.shape)

        if Data_parameters['corp']["state"] == True:
            if Data_parameters['corp']["random"]["state"] == False:
                # 按region类型分类    +++++++++++++++++++++++++++++++++++++++++++
                self.row_lines = Data_parameters['corp']["region"][region]["row_lines"]
                self.col_lines = Data_parameters['corp']["region"][region]["col_lines"]

            elif Data_parameters['corp']["random"]["state"] and mode == "img":
                row_random = Data_parameters['corp']["random"]["row_random"]
                row_random = int(np.random.uniform(-row_random,row_random))
                col_random = Data_parameters['corp']["random"]["col_random"]
                col_random = int(np.random.uniform(-col_random,col_random))
                row_lines = Data_parameters['corp']["row_lines"]
                col_lines = Data_parameters['corp']["col_lines"]
                self.row_lines = np.array(row_lines)+row_random
                self.col_lines = np.array(col_lines)+col_random
                print("corp:::",self.row_lines,self.col_lines)
            img = img[self.row_lines[0]:self.row_lines[1],self.col_lines[0]:self.col_lines[1]]

        if Data_parameters['flip']["state"] == True:
            if Data_parameters['flip']["random"]["state"]==False:
                self.flip = Data_parameters['flip']["flip"]
            if Data_parameters['flip']["random"]["state"] and mode == "img":
                self.flip = int(np.random.uniform(-2,2))
                print("flip:::",self.flip)
            img = cv2.flip(img, self.flip)

        if Data_parameters['rotate']["state"] == True:            
            rotation_range = Data_parameters['rotate']["rotation_range"]
            cols,rows = img.shape[1],img.shape[0]
            # 这里的第一个参数为旋转中心，第二个为旋转角度，第三个为旋转后的缩放因子# 可以通过设置旋转中心，缩放因子，以及窗口大小来防止旋转后超出边界的问题 第三个参数是输出图像的尺寸中心
            # 非90n的旋转角度，xml需要后续调整
            M=cv2.getRotationMatrix2D((cols/2,rows/2),rotation_range,1)
            img=cv2.warpAffine(img,M,(cols,rows),borderValue=(255,)*nums)
            print("rotate:::",rotation_range)

        if  Data_parameters['warp_aﬃne']["state"] == True:
            print("warp_aﬃne")
            cols,rows = img.shape[1],img.shape[0]
            pts1=np.float32(Data_parameters['warp_aﬃne']["pts1"])
            pts2=np.float32(Data_parameters['warp_aﬃne']["pts2"]) 
            M=cv2.getAffineTransform(pts1,pts2) 
            img=cv2.warpAffine(img,M,(cols,rows),borderValue=(255,)*nums)

        if  Data_parameters['warpPerspective']["state"] == True:
            print("warpPerspective")
            img = img.copy()
            cols,rows = img.shape[1],img.shape[0]
            pts1=np.float32(Data_parameters['warpPerspective']["pts1"])
            pts2=np.float32(Data_parameters['warpPerspective']["pts2"]) 
            M=cv2.getPerspectiveTransform(pts1,pts2) 
            img=cv2.warpPerspective(img,M,(cols,rows),borderValue=(255,)*nums)

        if Data_parameters['output_resize']["state"] == True:
            resize_size = Data_parameters['output_resize']["output_resize_size"]
            img = cv2.resize(img, (resize_size[0], resize_size[1]),interpolation=cv2.INTER_NEAREST)
            img = np.reshape(img,(resize_size[1], resize_size[0],nums))
            print("output_resize:::",img.shape)

        #################### 确保输出图片未WHC三通道 ################################
        img = np.reshape(img, (img.shape[0], img.shape[1], nums))
        print("output_resize:::", img.shape)

        ################### 测试 ################################
        # if mode =="xml":
        #     for i in range(nums):
        #         plt.imshow(img[:,:,i])
        #         plt.show()
        return img


def main():
    """
    支持 切割、旋转、水平垂直翻转。
    """
    ################################路径################################
    base= Data_parameters['data_dir']
    output = Data_parameters['save_dir']
    img_pieces_path = os.path.join(base,output,"img")
    xml_pieces_path = os.path.join(base,output,"xml") 

    if not os.path.exists(img_pieces_path):
        os.makedirs(img_pieces_path)
    if not os.path.exists(xml_pieces_path):
        os.makedirs(xml_pieces_path)

    ################################
    epochs = Data_parameters['epochs']
    for e in range(epochs):
        epoch = str(e+1)
        img_list = os.listdir(os.path.join(base,"img"))
        img_len = len(img_list)
        for i,b in enumerate(img_list):
            try:
                name = os.path.splitext(b)[0]
                img_mark = region_class(name)
                xml_path = os.path.join(base,"xml",name+".xml")
                img_path = os.path.join(base,'img',name+".jpg")
                # _name = name+"_"+epoch
                _name = name
                _xml_path = os.path.join(base,output,"xml",_name+".xml")
                _img_path = os.path.join(base,output,"img",_name+".jpg")
                ####################### img ###############################
                print('[{}/{}] --------------> {} ------------->'.format(i + 1, img_len, _name))
                img = cv2.imread(img_path)
                preprocess = Preprocess()
                img_cut = preprocess.preprocess(img = img,mode="img",region = img_mark)
                cv2.imwrite(_img_path,img_cut)
                
            except Exception:
                print("图片处理错误",img_path)
            try:
                ####################### xml ################################
                parsexml = et.parse(xml_path) 
                pic_xml = xml_to_pic(parsexml)
                pic_xml_cut = preprocess.preprocess(img = pic_xml,mode="xml",region = img_mark)
                xml = pic_to_xml(parsexml,pic_xml_cut,_name)
                xml.write(_xml_path, encoding='utf-8')
                
            except Exception:
                print("xml处理错误or无对应xml",xml_path)
    print("done!")

if __name__ == "__main__":
    # fire.Fire(main)
    main()
