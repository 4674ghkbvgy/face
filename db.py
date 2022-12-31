import sqlite3

# 连接数据库
conn = sqlite3.connect('face.db')
c = conn.cursor()

# 执行 SQL 语句创建表
c.execute("CREATE TABLE faces (id INTEGER PRIMARY KEY AUTOINCREMENT, face_data BLOB NOT NULL, name TEXT NOT NULL)")

# 提交事务
conn.commit()

# 关闭数据库连接
conn.close()