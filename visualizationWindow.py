from screenGuidedTrainingWindow import CollectionWorker
from utils import PageWindow
import matplotlib.pyplot as plt
from biomedical_hardware_readers import DelsysTrignoReader
from PyQt5.QtCore import QThread, QObject, pyqtSignal

# Need to do this on a thread
class displayWorker(QObject):
    finished = pyqtSignal()
    def __init__(self, vw, sensor_num):
        super(displayWorker, self).__init__()
        self.vw = vw
        self.signal_num = sensor_num

    def show_signals(self):
        self.run = True
        while self.run:
            self.emg = self.vw.basewindow.device['emg_buf'].read_matrix()
            plt.plot(self.emg[:,self.signal_num + 1]) # plot selected sensor
            plt.draw()
            plt.pause(0.1)
            plt.clf()

            if not self.vw.basewindow.state: 
                self.run = False
        
        plt.close()
        self.finished.emit()

class VisualizationWindow(PageWindow):
    def __init__(self, basewindow):
        super().__init__()
        # Lets store the handle to the settings window as a 
        self.basewindow = basewindow

        self.sensor_num = self.basewindow.vis_vars

    def display(self):
        self.make_parallel_worker()

    def make_parallel_worker(self,):
        self.thread = QThread() # define a thread
        self.worker = displayWorker(self, self.sensor_num) # define a worker
        self.worker.moveToThread(self.thread) # assign the worker to thread
        self.thread.started.connect(self.worker.show_signals)
        self.worker.finished.connect(self.thread.quit) # close the thread when the worker is done
        self.worker.finished.connect(self.worker.deleteLater) # when worker is done kill worker
        self.thread.finished.connect(self.thread.deleteLater) # when worker is done kill thread
        # start the thread
        self.thread.start()


        

