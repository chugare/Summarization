import pymysql


def connect_db(name):
    return pymysql.connect(host='172.19.241.222',
                           port=3306,
                           user='root',
                           password='1234',
                           database=name,
                           charset='utf8')

def get_by_source(source):
    con = connect_db('news2016')
    cur = con.cursor()


    try:
        cur.execute("select * from news_obj where source=\'%s\'" % source)

        res = cur.fetchall()
        con.commit()
        return res
    except:
        print("[W] some data fetch failed, but process continue")
    finally:
        cur.close()
        con.close()

# def create_table():
#
#     con = connect_db()
#     cur = con.cursor()
#     try:
#         sql = "create "
