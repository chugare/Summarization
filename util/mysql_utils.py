import pymysql


def connect_db():
    return pymysql.connect(host='172.19.241.222',
                           port=3306,
                           user='root',
                           password='1234',
                           database='news_2016',
                           charset='utf-8')

# def create_table():
#
#     con = connect_db()
#     cur = con.cursor()
#     try:
#         sql = "create "
