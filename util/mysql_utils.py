import pymysql


def connect_db():
    return pymysql.connect(host='172.19.241.222',
                           port=3306,
                           user='root',
                           password='1234',
                           database='news_2016',
                           charset='utf-8')


def dt_write():
    con = connect_db()
    cur = con.cursor()
    try:
        sql_str = ("INSERT INTO t_forward_file (Ffile_name, Ffile_md5) VALUES ('%s', '%s')" % (file_name, file_md5))
        cur.execute(sql_str)
        con.commit()
    except:
        con.rollback()
        logging.exception('Insert operation error')
        raise
    finally:
        cur.close()
        con.close()