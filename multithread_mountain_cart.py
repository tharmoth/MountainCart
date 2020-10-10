import queue
import threading
import time
from q_learning import QLearning
import numpy as np
import matplotlib.pyplot as plt

exitFlag = 0


class myThread(threading.Thread):
    def __init__(self, threadID, name, q, graph):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.q = q
        self.graph = graph

    def run(self):
        print("Starting " + self.name)
        process_data(self.name, self.q, self.graph)
        print("Exiting " + self.name)


def process_data(threadName, q, graph):
    while not exitFlag:
        queueLock.acquire()
        if not workQueue.empty():
            data = q.get()
            queueLock.release()
            # print("%s processing %s" + threadName)
            data.train(500)
            queueLock.acquire()
            graph.graph += data.convergence_graph / graph.number
            queueLock.release()
        else:
            queueLock.release()
        time.sleep(1)


class GraphHolder:
    def __init__(self):
        self.graph = np.zeros(50)
        self.number = 100


threadList = []
nameList = []
graph = GraphHolder()

for i in range(graph.number):
    threadList.append("Thread-" + str(i))
    nameList.append(QLearning("MountainCar-v0"))

queueLock = threading.Lock()
workQueue = queue.Queue()
threads = []
threadID = 1


# Create new threads
for tName in threadList:
    thread = myThread(threadID, tName, workQueue, graph)
    thread.start()
    threads.append(thread)
    threadID += 1

# Fill the queue
queueLock.acquire()
for word in nameList:
    workQueue.put(word)
queueLock.release()

# Wait for queue to empty
while not workQueue.empty():
    pass

# Notify threads it's time to exit
exitFlag = 1

# Wait for all threads to complete
for t in threads:
    t.join()
print("Exiting Main Thread")

plt.plot(graph.graph, label=type(QLearning).__name__)
plt.show()
