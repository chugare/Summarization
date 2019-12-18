
from util.mysql_utils import connect_db


class MysqlWriter():
    def __init__(self, buffer_size=1000):
        self.buffer = []
        self.buffer_size = buffer_size
        self.count = 0

        self.con = connect_db()
    def write(self, value):
        if len(self.buffer) >= self.buffer_size:
            self.dt_write(self.buffer)
            self.buffer = []
            self.count += self.buffer_size
        self.buffer.append(value)

    def close(self):
        if self.buffer:
            self.dt_write(self.buffer)
            self.count += len(self.buffer)
        self.con.close()
    def dt_write(self,values):
        cur = self.con.cursor()
        title, source, content = zip(*values)
        length = [len(i) for i in content]
        try:
            cur.executemany("INSERT INTO news_obj (title, source, length, content) VALUES ('%s', '%s','%s','%s')",[title,source,length,content])
            self.con.commit()
        except:
            self.con.rollback()
            raise
        finally:
            cur.close()