#!/usr/bin/python3
# -*- coding: utf-8 -*-

import io
import sys, os
import pandas as pd
import numpy as np
from PIL import Image, ImageQt, ImageOps
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn

from outputUI import Ui_MainWindow
from PaintBoard import PaintBoard

from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QApplication, QFileDialog
from PyQt5.QtWidgets import QLabel, QMessageBox, QPushButton, QFrame
from PyQt5.QtGui import QPainter, QPen, QPixmap, QColor, QImage
from PyQt5.QtCore import Qt, QPoint, QSize, QByteArray, QBuffer


MODE_MNIST = 1  # 本地输入
MODE_WRITE = 2  # 鼠标手写输入

# 读取训练数据和测试数据
train_data = pd.read_csv('./datasets/optdigits.tra', header=None)
test_data = pd.read_csv('./datasets/optdigits.tes', header=None)

# 将数据和标签分开
X_train = train_data.iloc[:,:-1].values
y_train = train_data.iloc[:,-1].values
X_test = test_data.iloc[:,:-1].values
y_test = test_data.iloc[:,-1].values

#归一化
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X1_train = scaler.fit_transform(X_train)

X2_train = np.zeros_like(X1_train)
#二值化
def binary_threshold_1d(image_array, threshold=127):
    # 进行二值化，像素值大于阈值的设为1，小于等于阈值的设为0
    binary_image = np.where(image_array > threshold, 1, 0)

    return binary_image

for i in range (len(X_train)):
    image_array = X_train[i]
    image = binary_threshold_1d(image_array, 7)

    X2_train[i] = image

print(X2_train[0].reshape(8,8))

# 定义 CNN 模型
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(64 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 1, 8, 8)  # 将数据重塑为图像格式
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        # print(x.shape)
        x = x.view(-1, 64 * 2 * 2)
        # print(x.shape)
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        # 初始化参数
        self.model = 1
        self.mode = MODE_MNIST
        self.result = [0, 0]

        # 初始化UI
        self.setupUi(self)
        self.center()

        # 初始化画板
        self.paintBoard = PaintBoard(self, Size=QSize(224, 224), Fill=QColor(0, 0, 0, 0))
        self.paintBoard.setPenColor(QColor(0, 0, 0, 0))
        self.dArea_Layout.addWidget(self.paintBoard)

        self.clearDataArea()

    # 窗口居中
    def center(self):
        # 获得窗口
        framePos = self.frameGeometry()
        # 获得屏幕中心点
        scPos = QDesktopWidget().availableGeometry().center()
        # 显示到屏幕中心
        framePos.moveCenter(scPos)
        self.move(framePos.topLeft())

    # 窗口关闭事件
    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message',
                                     "确定要退出吗?", QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.Yes)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

            # 清除数据待输入区

    def clearDataArea(self):
        self.paintBoard.Clear()
        self.lbDataArea.clear()
        self.lbResult.clear()
        self.lbCofidence.clear()
        self.result = [0, 0]

    """
    回调函数
    """

    # 模式下拉列表回调
    def cbBox_Mode_Callback(self, text):
        if text == '1：从本地获取图像':
            self.mode = MODE_MNIST
            self.clearDataArea()
            self.pbtGetMnist.setEnabled(True)

            self.paintBoard.setBoardFill(QColor(0, 0, 0, 0))
            self.paintBoard.setPenColor(QColor(0, 0, 0, 0))

        elif text == '2：鼠标手写输入':
            self.mode = MODE_WRITE
            self.clearDataArea()
            self.pbtGetMnist.setEnabled(False)

            # 更改背景
            self.paintBoard.setBoardFill(QColor(0, 0, 0, 255))
            self.paintBoard.setPenColor(QColor(255, 255, 255, 255))

    def cbBox_Mode_Callback1(self, text):
        if text == '1：KNN':
            self.model = 1
            self.clearDataArea()

        elif text == '2：SVM(线性核)':
            self.model = 2
            self.clearDataArea()

        elif text == '3：SVM(高斯核)':
             self.model = 3
             self.clearDataArea()

        elif text == '4：SVM(多项式核)':
             self.model = 4
             self.clearDataArea()

        elif text == '5：随机森林':
             self.model = 5
             self.clearDataArea()
             
        elif text == '6：CNN':
             self.model = 6
             self.clearDataArea()
        
    # 数据清除
    def pbtClear_Callback(self):
        self.clearDataArea()

    def QImage2PILImage(self, img):
        buffer = QBuffer()
        buffer.open(QBuffer.ReadWrite)
        img.save(buffer, "PNG")
        pil_im = Image.open(io.BytesIO(buffer.data()))
        return pil_im

    # 识别
    def pbtPredict_Callback(self):
        __img, img_array = [], []  # 将图像统一从qimage->pil image -> np.array [1, 1, 8, 8]

        # 获取qimage格式图像
        if self.mode == MODE_MNIST:
            __img = self.lbDataArea.pixmap()  # label内若无图像返回None
            if __img == None:  # 无图像则用纯黑代替
                # __img = QImage(224, 224, QImage.Format_Grayscale8)
                __img = ImageQt.ImageQt(Image.fromarray(np.uint8(np.zeros([224, 224]))))
            else:
                __img = __img.toImage()
        elif self.mode == MODE_WRITE:
            __img = self.paintBoard.getContentAsQImage()

        print(type(__img))
        # 转换成pil image类型处理
        pil_img = self.QImage2PILImage(__img)
        pil_img = pil_img.resize((8, 8), Image.LANCZOS)
        if self.mode == MODE_MNIST:
            pil_img = ImageOps.invert(pil_img)  # 反转颜色

        pil_img.save('test.png')

        img_array = np.array(pil_img.convert('L')).reshape(8, 8)
        print(img_array)
        # img_array = np.where(img_array>0.5, 1, 0)

        #预测结果
        if(self.model == 1):
            __result = self.KNN(img_array)

        elif(self.model == 2):
            __result = self.linearsvm(img_array)

        elif (self.model == 3):
            __result = self.Gausisvm(img_array)

        elif (self.model == 4):
            __result = self.polysvm(img_array)

        elif (self.model == 5):
            __result = self.RandomForest(img_array)

        else :
            __result = self.CNN(img_array)
        print (__result)

        self.result[0] = __result  # 预测的数字

        self.lbResult.setText("%d" % (self.result[0]))

    def KNN(self, Image):
        knn = KNeighborsClassifier(n_neighbors = 1)
        knn.fit(X2_train, y_train)
        # 将待预测的图像数据转换为二维数组，保持与训练数据一致
        print(Image.shape)
        #归一化
        # image_array_2d = self.normalize_image(Image)

        #二值化
        image_array_2d = binary_threshold_1d(Image, 127)
        print(image_array_2d)
        image_array_2d = np.reshape(image_array_2d, (1, -1))


        # 使用 KNN 模型进行预测
        predicted_class = knn.predict(image_array_2d)

        print(predicted_class)
        # 输出预测结果
        print(f"The predicted class is: {predicted_class[0]}")
        return predicted_class[0]

    def linearsvm(self, Image):
        # 初始化SVM模型，使用线性核
        svm = SVC(kernel='linear', C = 1)
        # image_array_2d = self.normalize_image(Image)
        # 二值化
        image_array_2d = binary_threshold_1d(Image, 127)
        image_array_2d = np.reshape(image_array_2d, (1, -1))

        svm.fit(X2_train, y_train)

        y_pred = svm.predict(image_array_2d)

        return y_pred[0]

    def Gausisvm(self, Image):
        # 初始化SVM模型，使用高斯核
        svm = SVC(C = 10, gamma = 0.001)
        # image_array_2d = self.normalize_image(Image)
        # 二值化
        image_array_2d = binary_threshold_1d(Image, 127)
        print(image_array_2d)
        image_array_2d = np.reshape(image_array_2d, (1, -1))

        print(X2_train.shape)
        svm.fit(X2_train, y_train)

        y_pred = svm.predict(image_array_2d)

        print(y_pred)
        return y_pred[0]

    def polysvm(self, Image):
        # 初始化SVM模型，使用高斯核
        svm = SVC(kernel='poly', C = 1, degree = 3)
        # image_array_2d = self.normalize_image(Image)
        # 二值化
        image_array_2d = binary_threshold_1d(Image, 127)
        image_array_2d = np.reshape(image_array_2d, (1, -1))

        svm.fit(X2_train, y_train)

        y_pred = svm.predict(image_array_2d)

        return y_pred[0]

    def RandomForest(self, Image):
        params = {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
        rf_classifier = RandomForestClassifier(**params)

        # image_array_2d = self.normalize_image(Image)
        # 二值化
        image_array_2d = binary_threshold_1d(Image, 127)
        image_array_2d = np.reshape(image_array_2d, (1, -1))

        rf_classifier.fit(X2_train, y_train)

        y_pred = rf_classifier.predict(image_array_2d)

        return y_pred[0]

    def CNN(self, Image):
        # 1. 加载模型
        model = CNNModel()  # 创建模型实例
        model.load_state_dict(torch.load('params.pth'))  # 加载模型参数

        image = self.fit_transform(Image)
        image_tensor = torch.from_numpy(image).float()  # 转换为 PyTorch 的 float 张量

        # 添加 batch 维度
        image_tensor = image_tensor.unsqueeze(0)  # 在第 0 维增加一个维度作为 batch

        model.eval()  # 设置模型为评估模式
        with torch.no_grad():
            output = model(image_tensor)

        # 获取预测结果
        _, predicted = torch.max(output, 1)
        prediction = predicted.item()  # 获取预测的类别

        # 这里的 prediction 即为模型对图片的预测结果（0 到 9 的数字类别）
        print(f"The predicted class for the image is: {prediction}")
        return prediction

    #标准化
    def fit_transform(self, image_array):
        mean = np.mean(image_array)
        std = np.std(image_array)

        # 对图片进行标准化处理
        normalized_image = (image_array - mean) / std
        return normalized_image

    #归一化
    def normalize_image(self, image_array):
        # 将图像数组缩放到 [0, 1] 范围
        normalized_image = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
        return normalized_image

        # 本地获取
    def pbtGetMnist_Callback(self):
        self.clearDataArea()

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        defaultPath = r"D:\pytorch-pro\课设\image"  # 设置默认路径
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", defaultPath,
                                                  "Images (*.png *.jpg *.bmp)", options=options)
        if fileName:
            pil_image = Image.open(fileName)
            q_image = self.convert_pil_to_qimage(pil_image)
            # 将 QPixmap 调整大小并设置到 QLabel 中显示
            scaled_pixmap = q_image.scaled(300, 200, Qt.KeepAspectRatio)  # 调整大小为 300x200

            # 将PIL图像对象转换为QPixmap并在QLabel上显示
            self.lbDataArea.setPixmap(scaled_pixmap)

    # 转化一下从本地导入的图片
    def convert_pil_to_qimage(self, pil_img):
        data = pil_img.convert("RGBA").tobytes("raw", "RGBA")
        q_image = QPixmap.fromImage(QImage(data, pil_img.size[0], pil_img.size[1], QImage.Format_RGBA8888))
        return q_image

if __name__ == "__main__":
    app = QApplication(sys.argv)
    Gui = MainWindow()
    Gui.show()

    sys.exit(app.exec_())