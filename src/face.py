import cv2
import dlib
import tkinter as tk
import sqlite3
import PIL.Image
import PIL.ImageTk
import numpy as np
import pandas as pd
# Tkinter美化
from tkinter import ttk
from ttkbootstrap import Style

# 使用 dlib 的人脸识别模型
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
face_rec = dlib.face_recognition_model_v1(
    'dlib_face_recognition_resnet_model_v1.dat')
# 打开摄像头
cap = cv2.VideoCapture(0)

known_face_descriptors = dict()


# 创建用于录入人脸的函数
def record_face():

    # 获取输入的名字
    name = entry.get()
    print(name)
    # 创建数据库连接
    if name:
        # 读取人脸图像数据
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 使用 dlib 检测人脸
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = detector(gray)

        # 遍历检测到的人脸
        for face in faces:
            # 获取人脸的关键点
            landmarks = predictor(gray, face)

            # 计算人脸的特征向量
            face_descriptor = face_rec.compute_face_descriptor(
                frame, landmarks)

            # 将人脸的特征向量转换为二进制数据
            face_data = np.array(face_descriptor).tobytes()
            conn = sqlite3.connect('faces.db')
            cursor = conn.cursor()
            # 将人脸数据存储到数据库中
            cursor.execute(
                '''INSERT INTO faces (name, face_data) VALUES (?, ?)''',
                (name, face_data))
            conn.commit()
            # 关闭数据库连接
            conn.close()
    global known_face_descriptors
    known_face_descriptors = get_known_face_descriptors()


def get_matching_name(face_descriptor):
    if face_descriptor:
        matching_name = ''
        t = 3.0
        # 遍历所有已知的人的特征向量
        global known_face_descriptors
        for name, stored_descriptor in known_face_descriptors.items():
            if len(stored_descriptor) != 0:
                # 计算当前特征向量和已知的特征向量的差值
                difference = np.abs(stored_descriptor - face_descriptor).sum()
                # 如果差值小于 0.6，则表示匹配成功
                if difference < t:
                    t = difference
                    # 记录匹配成功的名字
                    matching_name = name
                    if difference < 1.5:
                        break
        # 返回匹配成功的名字
        return matching_name, t


#在数据库中获取所有已知的特征向量
def get_known_face_descriptors():
    # 创建连接
    conn = sqlite3.connect('faces.db')
    # 创建游标
    cursor = conn.cursor()
    # 查询数据库
    cursor.execute('''SELECT name, face_data FROM faces''')
    # 获取结果
    results = cursor.fetchall()
    # 关闭连接
    conn.close()
    # 将结果保存到字典中
    known_face_descriptors = {
        name: np.frombuffer(face_data, np.float64)
        for name, face_data in results
    }
    # 返回结果
    return known_face_descriptors


def is_real_face(face_image):
    # # 将图像转换为灰度图像
    # gray_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

    # 计算傅里叶变换
    fft = np.fft.fft2(face_image)
    # 检测频率分布是否有较强的水平或竖直方向的倾斜
    # 如果有，则该图像可能是一张人脸照片（平面）
    # 否则，该图像可能是真实人脸（立体）
    # 获取幅值
    fft_magnitude = np.abs(fft)

    # 获取第一行和第一列的幅值
    first_row_magnitude = fft_magnitude[0]
    first_column_magnitude = fft_magnitude[:, 0]

    # 如果第一行和第一列的幅值均小于 5，则判断人脸图像是立体的
    if np.max(first_row_magnitude) < 5 and np.max(first_column_magnitude) < 5:
        print("!!!!!!!!!!")
        return False
    else:
        print("row:", np.max(first_row_magnitude), "column:",
              np.max(first_column_magnitude))

        return True


def detect_face():
    # 读取人脸图像数据
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 使用 dlib 检测人脸
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = detector(gray)

    # 遍历检测到的人脸
    for face in faces:
        # 获取人脸的关键点
        landmarks = predictor(gray, face)

        # 计算人脸的特征向量
        face_descriptor = face_rec.compute_face_descriptor(frame, landmarks)

        result, n = get_matching_name(face_descriptor)
        is_real_face(gray)
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 获取匹配的置信度
        if result:
            # 在图片上绘制矩形框，并显示人名
            str = result.__str__()
            cv2.putText(
                frame,
                str + ",n=" + n.__str__()[:4],
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            # else:
            # cv2.putText(frame, "unknow", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,)

    # 将图片转换为 PIL 图像
    image = PIL.Image.fromarray(frame)

    # 将 PIL 图像转换为 Tkinter 图像
    image = PIL.ImageTk.PhotoImage(image)

    # 将图像显示在 Label 中
    canvas.create_image(0, 0, image=image, anchor=tk.NW)
    canvas.image = image
    root.after(30, detect_face)


# def detect_face():
#     global known_face_descriptors
#     known_face_descriptors=get_known_face_descriptors()
#     while True:
#         # 读取人脸图像数据
#         ret, frame = cap.read()
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         # 使用 dlib 检测人脸
#         gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#         faces = detector(gray)

#         # 遍历检测到的人脸
#         for face in faces:
#             # 获取人脸的关键点
#             landmarks = predictor(gray, face)

#             # 计算人脸的特征向量
#             face_descriptor = face_rec.compute_face_descriptor(
#                 frame, landmarks)

#             result,n = get_matching_name(face_descriptor)

#             x1, y1 = face.left(), face.top()
#             x2, y2 = face.right(), face.bottom()
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#             # 获取匹配的置信度
#             if result:
#                 # 在图片上绘制矩形框，并显示人名
#                 str = result.__str__()
#                 cv2.putText(
#                     frame,
#                     str + ",n=" + n.__str__()[:4],
#                     (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.8,
#                     (0, 255, 0),
#                     2,
#                 )
#                 # else:
#                 # cv2.putText(frame, "unknow", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,)

#         # 将图片转换为 PIL 图像
#         image = PIL.Image.fromarray(frame)

#         # 将 PIL 图像转换为 Tkinter 图像
#         image = PIL.ImageTk.PhotoImage(image)

#         # 将图像显示在 Label 中
#         canvas.create_image(0, 0, image=image, anchor=tk.NW)
#         canvas.image = image

#         cv2.imshow('frame', frame)
#         # 如果用户按下 'r' 键，调用录入人脸对应人名的函数
#         key = cv2.waitKey(1)
#         if key & 0xFF == ord('r'):
#             # 调用录入人脸对应人名的函数
#             record_face()

#         if key & 0xFF == ord('q'):
#             # 关闭数据库连接
#             # conn.close()
#             break


def clean_table():

    # 连接数据库
    conn = sqlite3.connect('faces.db')
    c = conn.cursor()

    c.execute("DELETE FROM faces")
    conn.commit()

    # 关闭数据库连接
    conn.close()
    global known_face_descriptors
    known_face_descriptors = get_known_face_descriptors()


if __name__ == '__main__':

    # 创建 Tkinter 界面
    style = Style(theme='darkly')
    root = style.master
    root.title('人脸识别')
    # 创建标签
    label = tk.Label(
        root,
        text="使用说明:\n先点击识别人脸,如果需要录入人脸在识别窗口中摁”q“键,之后返回主界面输入名字,再点击录入人脸",
        wraplength=200)
    # 将标签放在窗口中
    label.pack()
    # # 创建显示人脸的画布
    canvas = tk.Canvas(root, width=640, height=480)
    canvas.pack()
    label = tk.Label(root, text='输入名字')
    label.pack(side='left')
    entry = tk.Entry(root, width=20)
    entry.pack(side='left')

    # 刷新 Tkinter 窗口
    root.update()

    # 创建“录入人脸”按钮
    record_button = ttk.Button(root,
                               text='录入人脸',
                               style='success.Outline.TButton',
                               command=record_face)
    record_button.pack(side='left', padx=5, pady=10)

    btn = ttk.Button(root,
                     text="清空数据库",
                     style='success.TButton',
                     command=clean_table)
    btn.pack(side='right', padx=5, pady=10)

    btn1 = ttk.Button(root,
                      text="获取人脸",
                      style='success.TButton',
                      command=detect_face)
    btn1.pack(side='right', padx=5, pady=10)
    # 开始检测人脸
    detect_face()
    root.mainloop()