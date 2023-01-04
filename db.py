import sqlite3

# 连接数据库
conn = sqlite3.connect('faces.db')
c = conn.cursor()

# # 执行 SQL 语句创建表
# c.execute("CREATE TABLE faces (id INTEGER PRIMARY KEY AUTOINCREMENT, face_data BLOB NOT NULL, name TEXT NOT NULL)")
c.execute('''CREATE TABLE IF NOT EXISTS faces (
    name text,
    face_data blob
)''')
# # 提交事务
# conn.commit()

# # 关闭数据库连接
# conn.close()

#执行 SQL 语句查询所有记录
c.execute("SELECT * FROM faces")


# c.execute("DELETE FROM faces")
# conn.commit()

# c.execute('''DROP TABLE faces;''')
# conn.commit()

# c.execute('''select * from sqlite_master where type="table";''')

# 获取所有记录
records = c.fetchall()

# 遍历记录列表，打印每条记录
for record in records:
    print(record)

# 关闭数据库连接
conn.close()