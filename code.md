# 2019.4.5
# BJTU-QJS-乔建森
# opencv python 制作图片数据集
# https://www.cnblogs.com/ZhengPeng7/p/7390119.html
import cv2
import numpy as np
import os
import shutil
from math import fabs, sin, cos, radians
from scipy.stats import mode

# 旋转并填充颜色
def get_img_rot_broa(img, degree, filled_color):
    """
    Desciption:
            Get img rotated a certain degree,
        and use some color to fill 4 corners of the new img.
    """

    # 获取旋转后4角的填充色
    if filled_color == -1:
        filled_color = mode([img[0, 0], img[0, -1],
                             img[-1, 0], img[-1, -1]]).mode[0]
    if np.array(filled_color).shape[0] == 2:
        if isinstance(filled_color, int):
            filled_color = (filled_color, filled_color, filled_color)
    else:
        filled_color = tuple([int(i) for i in filled_color])

    height, width = img.shape[:2]

    # 旋转后的尺寸
    height_new = int(width * fabs(sin(radians(degree))) +
                     height * fabs(cos(radians(degree))))
    width_new = int(height * fabs(sin(radians(degree))) +
                    width * fabs(cos(radians(degree))))

    mat_rotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

    mat_rotation[0, 2] += (width_new - width) / 2
    mat_rotation[1, 2] += (height_new - height) / 2

    # Pay attention to the type of elements of filler_color, which should be
    # the int in pure python, instead of those in numpy.
    img_rotated = cv2.warpAffine(img, mat_rotation, (width_new, height_new),
                                 borderValue=filled_color)
    # 填充四个角
    mask = np.zeros((height_new + 2, width_new + 2), np.uint8)
    mask[:] = 0
    seed_points = [(0, 0), (0, height_new - 1), (width_new - 1, 0),
                   (width_new - 1, height_new - 1)]
    for i in seed_points:
        cv2.floodFill(img_rotated, mask, i, filled_color)

    return img_rotated

# 处理后图片大小设置
image_size_x = 500
image_size_y = 500

# 图片名
name = 'pouch'
# 源文件路径
source_path = './pouch/'

# 保存图片的路径
savedpath = './pouchX/'

isExists = os.path.exists(savedpath)
if not isExists:
    os.makedirs(savedpath)
    print('path of %s is build' % (savedpath))

else:
    # 先删除
    shutil.rmtree(savedpath)
    # 再新建
    os.makedirs(savedpath)
    print('path of %s already exist and rebuild' % (savedpath))

image_list = os.listdir(source_path)      #获得文件名

i = 0
for file in image_list:
    for angle in range(-10, 11, 1):
        for multiple in np.arange(0.5, 4, 0.5):
            savedname = name + '_' + str(i) + '_' + str(angle) + '_' + str(multiple)+ '.jpg'
            # 加载图片，转成灰度图，换成自己的图片
            image = cv2.imread(source_path+file)
            imag = get_img_rot_broa(image,angle,-1)
            if multiple>=1:
                imagm = cv2.resize(imag,None,fx=multiple,fy=multiple,interpolation=cv2.INTER_LINEAR)
            else:
                imagm = cv2.resize(imag,None,fx=multiple,fy=multiple,interpolation=cv2.INTER_AREA)
            # 修改尺寸
            # imagcut = cv2.resize(imag, (image_size_x, image_size_y), 0, 0, cv2.INTER_LINEAR)
            # cv2.imwrite(savedpath + savedname, imagcut)
            cv2.imwrite(savedpath + savedname, imagm)
            # 重命名并且保存
            print('image of %s is saved' % (savedname))

    i = i + 1

print("批量处理完成——旋转缩放")

# 源文件路径
source_path = './pouchX/'
file_dir = source_path
# 保存图片的路径
savedpath = './pouchXcut/'

isExists = os.path.exists(savedpath)
if not isExists:
    os.makedirs(savedpath)
    print('path of %s is build' % (savedpath))

else:
    # 先删除
    shutil.rmtree(savedpath)
    # 再新建
    os.makedirs(savedpath)
    print('path of %s already exist and rebuild' % (savedpath))

image_list = os.listdir(source_path)      #获得文件名
j = 0
for file in image_list:
    j = j + 1
    # 加载图片，转成灰度图，换成自己的图片
    image = cv2.imread(source_path+file)
    
    savedname = name + '_' + str(j) + '.jpg'          
    # 修改尺寸
    imagcut = cv2.resize(image, (image_size_x, image_size_y), 0, 0, cv2.INTER_LINEAR)
    cv2.imwrite(savedpath + savedname, imagcut)
    #cv2.imwrite(savedpath + savedname, imagm)
    # 重命名并且保存
    print('image of %s is saved' % (savedname))
    
print("批量处理完成——裁剪重命名")







