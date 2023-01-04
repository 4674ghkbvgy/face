import cv2
import tkinter as tk
import sqlite3

# 加载 Haar 特征分类器
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 使用 Tkinter 创建窗口和按钮
root = tk.Tk()
root.title("人脸识别")
root.geometry("400x400")

#使用 Tkinter 创建输入框
name_entry = tk.Entry(root)
name_entry.pack()

#定义录入人脸对应人名的函数
def record_name(face_data):
    # 获取用户输入的人名
    name = name_entry.get()
    print(name)
    # 连接数据库
    conn = sqlite3.connect('face.db')
    c = conn.cursor()
    # 使用 SQL 语句将人脸数据和人名存储到数据库中
    c.execute("INSERT INTO faces (face_data, name) VALUES (?, ?)", (face_data, name))
    # 提交事务
    conn.commit()
    # 关闭数据库连接
    conn.close()
xxx

def get_face():
    # 使用摄像头捕获人脸图像
    cap = cv2.VideoCapture(0)
    conn = sqlite3.connect('face.db')
    c = conn.cursor()
    while True:
        # 读取摄像头中的帧
        ret, frame = cap.read()
        
        # 使用 Haar 特征分类器检测人脸
        # faces = face_cascade.detectMultiScale(frame, 1.3, 5)

        # faces = face_cascade.detectMultiScale(frame, 1.3, 2)
        
        faces = face_cascade.detectMultiScale(frame, 1.3, 2)
        # 使用 OpenCV 将人脸图像转换为字符串
        success, encoded_image = cv2.imencode('.jpg', frame)

        # 将字符串转换为 bytes 类型
        face_data = encoded_image.tobytes()

        conn = sqlite3.connect('face.db')
        c = conn.cursor()
        # 使用 SQL 语句查询人脸数据对应的人名
        c.execute("SELECT name FROM faces WHERE face_data=?", (face_data,))
        name = c.fetchone()
        print("Name:",name)
        conn.close()  
        # 在图像中画出人脸矩形
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, name, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        # 显示人脸图像
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        
        # 如果用户按下 'r' 键，调用录入人脸对应人名的函数
        if key & 0xFF == ord('r'):
           
            # 调用录入人脸对应人名的函数
            record_name(face_data)

        if key & 0xFF == ord('q'):
            # 关闭数据库连接                      
            break




btn = tk.Button(root, text="获取人脸", command=get_face)
btn.pack()
    #运行 Tkinter 程序
root.mainloop()