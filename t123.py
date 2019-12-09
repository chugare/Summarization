import random, time
from threading import Condition, Thread,RLock
condition = RLock()
box = []
def producer(box, nitems):
    for i in range(nitems):
        # time.sleep(random.randrange(2, 5))  # Sleeps for some time.
        condition.acquire()
        num = random.randint(1, 10)
        box.append(num)  # Puts an item into box for consumption.
        # condition.notify()  # Notifies the consumer about the availability.
        print("Produced:", num)
        condition.release()
def consumer(box, nitems):
    for i in range(nitems):
        condition.acquire()
        # condition.wait()  # Blocks until an item is available for consumption.
        print("%s: Acquired: %s" % (time.ctime(), box.pop()))
        condition.release()
threads = []
nloops = random.randrange(3, 6)
for func in [producer, consumer]:
    threads.append(Thread(target=func, args=(box, nloops)))
    threads[-1].start()  # Starts the thread.
for thread in threads:
    thread.join()
print("All done.")