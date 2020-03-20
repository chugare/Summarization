
from util.mysql_utils import connect_db


class MysqlWriter():
    def __init__(self, buffer_size=1000):
        self.buffer = []
        self.buffer_size = buffer_size
        self.count = 0

        self.con = connect_db('lcsts')
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
        id,title, content = zip(*values)

        conlength = [len(i) for i in content]
        sumlength = [len(i) for i in title]

        val = zip(id,title,tuple(sumlength),content,tuple(conlength))
        try:
            cur.executemany("INSERT INTO base (id, summary, sumlen, content,conlen) VALUES (%s, %s, %s, %s ,%s)", val )
            res = self.con.commit()
        except Exception as e:
            print("[W] some data insert failed %s, but process continue"%e.args[1])
            # raise e
        finally:
            cur.close()




