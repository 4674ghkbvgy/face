import cv2
import dlib
import tkinter as tk
import sqlite3
from PIL import Image, ImageTk
import numpy as np




# 使用 dlib 的人脸识别模型
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
face_rec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
# 创建 Tkinter 界面
root = tk.Tk()
root.title('人脸识别')

# # 创建显示人脸的画布
# canvas = tk.Canvas(root, width=640, height=480)
# canvas.pack()
label = tk.Label(root, text='输入名字')
label.pack(side='left')
entry = tk.Entry(root, width=20)
entry.pack(side='left')
# 打开摄像头
cap = cv2.VideoCapture(0)

# 创建用于录入人脸的函数
def record_face():

        # 获取输入的名字
        name = entry.get()
        print (name)
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
                face_descriptor = face_rec.compute_face_descriptor(frame, landmarks)

                # 将人脸的特征向量转换为二进制数据
                face_data = np.array(face_descriptor).tobytes()
                conn = sqlite3.connect('faces.db')
                cursor = conn.cursor()
                # 将人脸数据存储到数据库中
                cursor.execute('''INSERT INTO faces (name, face_data) VALUES (?, ?)''', (name, face_data))
                conn.commit()
                # 关闭数据库连接
                conn.close()

def compute_distance(face_data1, face_data2):
    return np.sqrt(np.sum((np.frombuffer(face_data1, np.float64) - np.frombuffer(face_data2, np.float64))**2))

def detect_face():
    while True:
        # 读取人脸图像数据
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 使用 dlib 检测人脸
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = detector(gray)
        # 创建数据库连接
        conn = sqlite3.connect('faces.db')
        cursor = conn.cursor()      
        # 遍历检测到的人脸
        for face in faces:
            # 获取人脸的关键点
            landmarks = predictor(gray, face)
            
            # 计算人脸的特征向量
            face_descriptor = face_rec.compute_face_descriptor(frame, landmarks)

            face_data = np.array(face_descriptor).tobytes()

            cursor.execute('''SELECT * FROM faces ORDER BY compute_distance(face_data - ?)''', (face_data,))
            result = cursor.fetchone()
            x1, y1 = face.left(), face.top()
            x2, y2 = face.right(), face.bottom()     
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)   
            # 获取匹配的置信度
            # fontStyle = ImageFont.truetype(
            # "/home/zty/project/moring_check/face/font/simsun.ttc", 32, encoding="utf-8")
            if result :
                n=np.abs(np.frombuffer(result[1], np.float64) - np.frombuffer(face_data, np.float64)).sum()
                # 在图片上绘制矩形框，并显示人名
                # if n<4:
                str=result[0].__str__()
                cv2.putText(frame, str+",n="+n.__str__()[:4], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,)
                # else:
                    # cv2.putText(frame, "unknow", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,)
            # # 将图片转换为 PIL 图像
            # image= Image.fromarray(frame)

            # # 将 PIL 图像转换为 Tkinter 图像
            # image = ImageTk.PhotoImage(image)

            # # 显示图像
            # label.config(image=image)
            # label.image = image

        cv2.imshow('frame', frame)
        #如果用户按下 'r' 键，调用录入人脸对应人名的函数
        key = cv2.waitKey(1)        
        if key & 0xFF == ord('r'):
            # 调用录入人脸对应人名的函数
            record_face()

        if key & 0xFF == ord('q'):
            # 关闭数据库连接
            conn.close()
            # 关闭数据库连接                      
            break
# 刷新 Tkinter 窗口
# root.update()
def clean_table():

    # 连接数据库
    conn = sqlite3.connect('faces.db')
    c = conn.cursor()

    c.execute("DELETE FROM faces")
    conn.commit()

    # 关闭数据库连接
    conn.close()

# 创建“录入人脸”按钮
record_button = tk.Button(root, text='录入人脸', command=record_face)
record_button.pack()

btn = tk.Button(root, text="获取人脸", command=detect_face)
btn.pack()
btn1 = tk.Button(root, text="清空数据库", command=clean_table)
btn1.pack()
# 开始检测人脸
# detect_face()

# 进入 Tkinter 消息循环
root.mainloop()