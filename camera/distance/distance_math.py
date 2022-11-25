import cv2
import numpy as np
import math
import os
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd


#相机读取照片大小，这里要和标定图片大小一致
win_width = 640
win_height = 480
mid_width = int(win_width / 2)
mid_height = int(win_height / 2)
real_wid = 297 ##此值不需要修改

capture = cv2.VideoCapture(0)
capture.set(3, win_width)
capture.set(4, win_height)
#标定后的相机内参
cameraMatrix = np.array([ [415.989442700871, 0, 325.631065891229], [0, 415.605441273097, 245.542575088416], [0, 0, 1]], dtype = np.float64 )
#标定后的相机畸变系数
distCoeffs = np.array([-0.343950902566815,0.202990690789230,-0.000562615515864942,-0.000728933790773100,-0.0872823864364995], dtype = np.float32)


res = []
flag = capture.isOpened()
mask = np.zeros((480, 640, 3 ),dtype=np.uint8)
mask[0:220,:,:] = 255

while(flag):
    ret, img = capture.read()
    if ret != True:
        continue
    # 读取矫正图片
    img_rectify = cv2.undistort(img ,cameraMatrix, distCoeffs)    
    # 遮盖
#     img_rectify = cv2.subtract(img_rectify,mask)# 如有天空干扰，可增加mask
    # 寻找轮廓
    gray = cv2.cvtColor(img_rectify, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, binary = cv2.threshold(gray, 250, 255, 0)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) >= 1:
    # 最大轮廓
        areas = []
        for c in range(len(contours)):
            areas.append(cv2.contourArea(contours[c]))
        # if len(areas) != 0:
        max_id = areas.index(max(areas))                    # 最大轮廓id
        max_rect = cv2.minAreaRect(contours[max_id])        # 最小外接矩形
        # 画出轮廓
        box =  cv2.boxPoints(max_rect)
        cv2.drawContours(img_rectify, [np.int0(box)], -1, (0, 255, 0), thickness=2)
        width =  max(max_rect[1])                           # 宽度，为离相机最近边长
        # 像素点坐标
        u = (max(box[:, 0]) + min(box[:, 0]) ) /2           # 横坐标
        box = sorted(box,key = lambda x:x[1], reverse = True)
        v = (box[0][1] + box[1][1])/2                       # 纵坐标
        # 画出像素坐标标点
        cv2.drawMarker(img_rectify, (int(u), int(v)), (0,  0, 255),cv2.MARKER_TILTED_CROSS)

        # 计算距离
        dis_mm = (real_wid * cameraMatrix[0, 0]) / width    # 相对x
        cv2.putText(img_rectify, "{:.2f}".format(dis_mm/1000.0), (int(u), int(v)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow("contours", img_rectify)
        k = cv2.waitKey(0) & 0xFF    
        #print("坐标:",u, v)
        #print("距离", dis_mm)
        if k == ord('s') :# 将需要的截图保存的res列表中
            res.append([u, v, dis_mm])# 结果都保存到res中
        elif  k == ord('q') :
            break
cv2.destroyAllWindows()


def file_check(file_name):
    temp_file_name = file_name
    i = 1
    while i:
        if os.path.exists("./" + temp_file_name):
            name, suffix = file_name.split('.')
            name += '(' + str(i) + ')'
            temp_file_name = name+'.'+suffix
            i = i+1
        else:
            return temp_file_name

# 保存res

columns = ["x","y","distance"]
df = pd.DataFrame(columns = columns, data = res)

filename = file_check("distance/csv/trash.csv")

df.to_csv(filename, encoding='utf-8')