import cv2
import dlib
import dlib
import cv2
import numpy as np
import sqlite3
import tkinter as tk
from PIL import Image, ImageTk

# 使用 dlib 的人脸识别模型
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
face_rec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# 连接数据库
conn = sqlite3.connect('faces.db')
cursor = conn.cursor()
cap = cv2.VideoCapture(0)

root = tk.Tk()
root.geometry('200x100')
label = tk.Label(root, text='输入名字')
label.pack(side='left')
entry = tk.Entry(root, width=20)
entry.pack(side='left')
    # # 创建 Tkinter 输入框
    # root = tk.Tk()
    # root.geometry('200x100')
    # label = tk.Label(root, text='输入名字')
    # label.pack(side='left')
    # entry = tk.Entry(root, width=20)
    # entry.pack(side='left')
    
    # # 创建确认按钮
    # def confirm():

# 创建用于录入人脸的函数
def record_face():

        # 获取输入的名字
        name = entry.get()
        
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

            # 将人脸的特征向量转换为二进制数据
            face_data = face_descriptor.tobytes()

            # 将人脸数据存储到数据库中
            cursor.execute('''INSERT INTO faces (name, face_data) VALUES (?, ?)''', (name, face_data))
            conn.commit()

            # 关闭数据库连接
            conn.close()

# # 关闭 Tkinter 窗口
# root.destroy()

# 创建用于检测人脸的函数
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

        # 将人脸的特征向量转换为二进制数据
        face_data = face_descriptor.tobytes()

        # 查询数据库中与当前图片最相近的特征
        cursor.execute('''SELECT * FROM faces ORDER BY ABS(face_data - ?)''', (face_data,))
        result = cursor.fetchone()

        # 如果查询到的特征与当前图片的特征的差值小于阈值，则认为是同一个人
        if result and np.abs(np.frombuffer(result[2], np.float64) - np.frombuffer(face_data, np.float64)).sum() < 0.6:
            # 在图片上绘制矩形框，并显示人名
            x1, y1 = face.left(), face.top()
            x2, y2 = face.right(), face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, result[1], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 将图片转换为 PIL 图像
        image= Image.fromarray(frame)

        # 将 PIL 图像转换为 Tkinter 图像
        image = ImageTk.PhotoImage(image)

        # 显示图像
        label.config(image=image)
        label.image = image
        # 刷新 Tkinter 窗口
        root.update()

# 循环检测人脸
root.after(30, detect_face)
# # 关闭摄像头
# cap.release()



# # 创建用于录入人脸数据的函数
# def record_face():
#     # 请求输入人脸对应的名字
#     name = input('请输入人脸对应的名字：')
    
#     # 读取人脸图像数据
#     ret, frame = cap.read()
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
#     # 使用 dlib 检测人脸
#     gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#     faces = detector(gray)
#     # 创建数据库连接
#     conn = sqlite3.connect('faces.db')
#     cursor = conn.cursor()
#     # 遍历检测到的人脸
#     for face in faces:
#         # 获取人脸的关键点
#         landmarks = predictor(gray, face)
#         # 遍历人脸的关键点
#         for point in landmarks.parts():
#             # 在图像上绘制关键点
#             cv2.circle(frame, (point.x, point.y), 2, (0, 255, 0), -1)
        
#         # 获取人脸图像
#         face_image = frame[face.top():face.bottom(), face.left():face.right()]
        
#         # 将人脸图像转换为二进制数据
#         face_data = cv2.imencode('.jpg', face_image)[1].tobytes()
        
#         # 将人脸数据插入数据库
#         cursor.execute('''INSERT INTO faces (name, face_data) VALUES (?, ?)''', (name, face_data))
#         conn.commit()

#     # 关闭数据库连接
#     conn.close()

# # 创建用于检测人脸的函数
# def detect_face():

#     while True:
#             # 读取人脸图像数据
#         ret, frame = cap.read()
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         # 使用 dlib 检测人脸
#         gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#         faces = detector(gray)

#             # 创建数据库连接
#         conn = sqlite3.connect('faces.db')
#         cursor = conn.cursor()
#         # 遍历检测到的人脸
#         for face in faces:
#             # 获取人脸的关键点
#             landmarks = predictor(gray, face)
#             # 遍历人脸的关键点
#             for point in landmarks.parts():
#                 # 在图像上绘制关键点
#                 cv2.circle(frame, (point.x, point.y), 2, (0, 255, 0), -1)
            
#             # 获取人脸图像
#             face_image = frame[face.top():face.bottom(), face.left():face.right()]
#             # 将人脸图像转换为二进制数据
#             face_data = cv2.imencode('.jpg', face_image)[1].tobytes()
#             # 查询人脸数据
#             cursor.execute('''SELECT name, face_data FROM faces WHERE face_data = ?''', (face_data,))
#             # 获取查询结果
#             result = cursor.fetchone()
#             if result:
#                 # 识别到的人脸名称
#                 name = result[0]
#             else:
#                 # 未识别到的人脸
#                 name = 'Unknown'
#             # 在图像上绘制人脸名称
#             cv2.putText(frame, name, (face.left(), face.top()), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     #     # 将图像转换为 Tkinter PhotoImage 对象
#     #     image = Image.fromarray(frame)
#     #     image = ImageTk.PhotoImage(image)
#     # # 更新画布中的图像
#     #     canvas.create_image(0, 0, image=image, anchor=tk.NW)
#     #     root.after(30, detect_face)
#         cv2.imshow('frame', frame)
#         #如果用户按下 'r' 键，调用录入人脸对应人名的函数
#         key = cv2.waitKey(1)        
#         if key & 0xFF == ord('r'):
#             # 调用录入人脸对应人名的函数
#             record_face()

#         if key & 0xFF == ord('q'):
#             # 关闭数据库连接
#             conn.close()
#             # 关闭数据库连接                      
#             break


import numpy as np

def compute_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))
face_descriptors = [face1, face2, face3, ...]

# 计算距离
distances = [compute_distance(face_descriptor, current_face_descriptor) for face_descriptor in face_descriptors]

# 排序
sorted_indexes = sorted(range(len(distances)), key=lambda k: distances[k])

# 将最小距离的特征放在最前面
sorted_face_descriptors = [face_descriptors[i] for i in sorted_indexes]