# Face Recognition

这是一个人脸识别项目。

## v1.0功能

- 人脸检测
- 人脸识别
- 对人脸进行命名
- 将命名后的人脸数据存入数据库
- 从数据库中快速调用人脸数据

### 存在的问题

- 人脸比较存在bug

### TODO

- 增加读取数据库中所有特征向量并和当前人脸特征向量比较的函数

## v2.0新增功能

- 简单的软件使用说明
- 操作界面优化
- 从数据库中快速调用人脸数据并识别人脸姓名


### v2.0演示视频


https://user-images.githubusercontent.com/72641410/210591526-5eed48ae-f512-4b1e-980b-04616ba9ce86.mp4

### 存在的问题

- 操作不够人性化，要两个界面来回切换

### TODO

- 增加生物识别，判断真假人的函数
- 界面实现简单多线程，以达到在一个界面操作的目的

## v3.0新增功能

- 生物识别，通过傅立叶变化的幅值判断是真人还是图片
- 操作界面多功能多线程并行操作

### 存在的问题

- 文件格式不够规范

## 正式版（v3.5）
### 演示视频
file:///home/zty/è§†é¢‘/2023-01-06 23-40-34.mp4



## 使用

需要安装 Python 3 和以下库：

- opencv-python
- dlib
- sqlite3
- tkinter

运行 `src/face.py` 文件即可开始使用。

```
 python3 ./src/face.py
```

## 许可证

本项目使用 [MIT 许可证](LICENSE)。
