from screenGuidedTrainingWindow import CollectionWorker
from utils import PageWindow
import matplotlib.pyplot as plt
import numpy as np
from PyQt5 import QtWidgets
from pyqtgraph import PlotWidget
import pyqtgraph as pg
from biomedical_hardware_readers import DelsysTrignoReader
from PyQt5.QtCore import QThread, QObject, pyqtSignal
from PyQt5.QtWidgets import QMainWindow, QMessageBox
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasAgg as FigureCanvas
import matplotlib.figure as mpl_fig
import matplotlib as mpl
import matplotlib.animation as anim
from time import sleep
import sys

# Need to do this on a thread
# class displayWorker(QObject):
#     finished = pyqtSignal()
#     def __init__(self, vw):
#         super(displayWorker, self).__init__()
#         self.vw = vw
#         #self.signal_num = sensor_num

#     def show_signals(self):
#         self.run = True
#         while self.run:
#             ## GET THE EMG DATA
#             self.emg = self.vw.basewindow.device['emg_buf'].read_matrix()
#             self.emg = self.emg[:,2:18]
#             self.active_channels = self.emg.sum(axis=0) != 0 # only want to plot the active channels
#             self.active_channels_data = self.emg[:, self.active_channels]
#             self.total_channel_num = len(self.active_channels_data[0])
#             self.active_channels_data = np.array(self.active_channels_data)
            
#             ## NOW PLOT THE DATA
#             for channel in range(self.total_channel_num):
#                 self.vw.plot_canvas.plot(self.active_channels_data[:, channel])
#                 self.active_channels_data += 10 
            
#             # plt.draw()
#             # plt.pause(0.1)
#             # plt.clf()

#             sleep(0.1)
#             self.vw.plot_canvas.clear()

#             if not self.vw.basewindow.state: 
#                 self.run = False
        
#                 plt.close()
#                 self.finished.emit()

# class VisualizationWindow(QMainWindow):
#     def __init__(self, basewindow):
#         super().__init__()
#         # Lets store the handle to the settings window as a 
#         self.basewindow = basewindow

#         self.plot_canvas = pg.PlotWidget()
#         self.setCentralWidget(self.plot_canvas)

#         ## CREATE A WINDOW TO PUT FIGURE ON
#         # self.setGeometry(500, 20, 750, 600)
#         # self.setWindowTitle("Real-time EMG signals")
#         # self.frame = QtWidgets.QFrame(self)
#         # self.BoxLayout = QtWidgets.QVBoxLayout()
#         # self.frame.setLayout(self.BoxLayout)
#         # self.setCentralWidget(self.frame)
                
#         # ## Place matplotlib figure
#         # self.Fig = createFigureCanvas()
#         # self.BoxLayout.addWidget(self.Fig)

#     def display(self):
#         ## DISPLAY WINDOW
#         #self.show()

#         ## MAKE A PARALLEL WORKER
#         self.make_parallel_worker()

#     def make_parallel_worker(self,):
#         self.thread = QThread() # define a thread
#         self.worker = displayWorker(self) # define a worker
#         self.worker.show_signals()
#         self.worker.moveToThread(self.thread) # assign the worker to thread
#         self.thread.started.connect(self.worker.show_signals)
#         self.worker.finished.connect(self.thread.quit) # close the thread when the worker is done
#         self.worker.finished.connect(self.worker.deleteLater) # when worker is done kill worker
#         self.thread.finished.connect(self.thread.deleteLater) # when worker is done kill thread
#         # start the thread
#         self.thread.start()

# class createFigureCanvas(FigureCanvas, anim.FuncAnimation):
#     finished = pyqtSignal()
#     def __init__(self, vw):
#         super().__init__(mpl.figure.Figure())
#         self.vw = vw # reference to VisualizationWindow

#         ## Create a reference to a figure
#         self.graph = self.figure.subplots()

#     def canvas_update(self): # need to get a timer to call this
#         self.graph.clear()
        
#         self.run = True
#         while self.run:
#             ## GET THE EMG DATA
#             self.emg = self.vw.basewindow.device['emg_buf'].read_matrix()
#             self.emg = self.emg[:,2:18]
#             self.active_channels = self.emg.sum(axis=0) != 0 # only want to plot the active channels
#             self.active_channels_data = self.emg[:, self.active_channels]
#             self.total_channel_num = len(self.active_channels_data[0])
#             self.active_channels_data = np.array(self.active_channels_data)
            
#             ## NOW PLOT THE DATA
#             for channel in range(self.total_channel_num):
#                 self.graph.plot(self.active_channels_data[:, channel])
#                 self.active_channels_data += 10 
            
#             self.draw()
#             time.sleep(0.1)
#             # plt.pause(0.1)
#             # plt.clf()

#             if not self.vw.basewindow.state: 
#                 self.run = False
#                 plt.close()
#                 self.finished.emit()

            
# ## LETS TRY THIS AGAIN
# class displayWorker(QObject):
#     finished = pyqtSignal()
#     def __init__(self, vw):
#         super(displayWorker, self).__init__()
#         self.vw = vw
#         self.run = True
#         self.active_channels_data = []

#     def get_signals(self):
#         while self.run:
#             self.emg = self.vw.basewindow.device['emg_buf'].read_matrix()
#             self.emg = self.emg[:,2:18]
#             self.active_channels = self.emg.sum(axis=0) != 0 # only want to plot the active channels
#             self.active_channels_data = self.emg[:, self.active_channels]
#             self.total_channel_num = len(self.active_channels_data[0])
#             self.active_channels_data = np.array(self.active_channels_data)
#             self.vw.plot_sigs(self.active_channels_data, self.total_channel_num)
            
#             # if self.vw.basewindow.state is None:
#             #     self.run = False
            
#             sleep(0.1)
#         ## STOP THE THREAD
#         self.finished.emit()

# class VisualizationWindow(QtWidgets.QMainWindow):
#     def __init__(self, basewindow):
#         super(VisualizationWindow, self).__init__()

#         # Lets store the handle to the settings window as a 
#         self.basewindow = basewindow

#         ## DEFINE PLOT CANVAS
#         self.plotWidget = pg.PlotWidget()
#         self.setCentralWidget(self.plotWidget)
#         #self.line = self.plotWidget.getPlotItem().plot()
#         self.plotWidget.setBackground('w')


#     def display(self):
#         try:
#             if not self.basewindow.state is None: # only if system is connected
#                 ## START STREAMING THE LOOP
#                 self.basewindow.device['reader'].start()
#                 self.basewindow.device['reader'].wait_for_reading_loop()
#                 self.basewindow.state.append('VISUALIZING')

#                 ## MAKE A PARALLEL WORKER
#                 self.make_parallel_worker()

#                 ## PLOT THE SIGNALS
#                 # while True:
#                 #     self.plot_sigs(self.worker.active_channels_data, self.worker.total_channel_num)
#                 #     sleep(0.2)


#         except:
#             msg = QMessageBox()
#             msg.setWindowTitle("Display Status")
#             msg.setText("Cannot visualize")
#             msg.setIcon(QMessageBox.Critical)
#             msg.setStandardButtons(QMessageBox.Ok)
#             msg.exec_() # shows the message box
#             self.basewindow.goto("main")

#     def make_parallel_worker(self):
#         self.thread = QThread() # define a thread
#         self.worker = displayWorker(self) # define a worker
#         #self.worker.get_signals()
#         self.worker.moveToThread(self.thread) # assign the worker to thread
#         self.thread.started.connect(self.worker.get_signals)
#         self.worker.finished.connect(self.thread.quit) # close the thread when the worker is done
#         self.worker.finished.connect(self.worker.deleteLater) # when worker is done kill worker
#         self.thread.finished.connect(self.thread.deleteLater) # when worker is done kill thread
#         # start the thread
#         self.thread.start()

#     def plot_sigs(self, emg_data, total_channel_num):
#         ## THIS IS ACTUALLY PLOTTING
#         self.emg_data = emg_data
#         self.total_channel_num = total_channel_num
        
#         self.plotWidget.clear()
#         for channel in range(self.total_channel_num):
#             self.plotWidget.plot(self.emg_data[:, channel])
#             self.emg_data += 1 
#         self.show()

#     def closeEvent(self, event):
#         ## STOP THE THREAD
#         self.worker.run = False

#         ## REMOVE VISUALIZING STATE 
#         self.basewindow.state.remove('VISUALIZING')

#         ## CLOSE THE BUFFERS
#         self.basewindow.device['reader'].shutdown()
#         self.basewindow.device['reader'].join()
#         self.basewindow.device['emg_buf'].close()
#         self.basewindow.device['aux_buf'].close()

#         ## NOW CLOSE THE WINDOW
#         event.accept()

# def main():
#     app = QtWidgets.QApplication(sys.argv)
#     main = VisualizationWindow()
#     #main.show()
#     sys.exit(app.exec_())


# if __name__ == '__main__':
#     main()
    


class VisualizationWindow(QMainWindow):
    def __init__(self, basewindow):
        super(VisualizationWindow, self).__init__()
        # window properties
        self.setObjectName("visualization_window")
        self.setGeometry(300, 300, 800, 600)

        ## Central Widget
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.setCentralWidget(self.centralwidget) 

        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.lyt =QtWidgets.QVBoxLayout()
        self.frame.setLayout(self.lyt)


        # Lets store the handle to the main window  
        self.basewindow = basewindow

    def onRender(self):
        self.create_figure()
        self.lyt.addWidget(self.figure)

    def create_figure(self):
        self.figure = createFigureCanvas(self.basewindow, x_len = 2000, data_range = [0,100], interval = 20)
    
class createFigureCanvas(FigureCanvas, anim.FuncAnimation):
    ## THIS IS WHERE THE PLOT WILL BE DRAWN
    def __init__(self, vw, x_len:int, data_range:list, interval:int) -> None:
        ## PARAMETERS:
        # x_len
        # data_range: list of 2 elements with the min at index 0 and max at index 1
        #
       
        self.vw = vw

        FigureCanvas.__init__(self, mpl_fig.Figure())

        # Store the variables
        self.x_len = x_len
        self.data_range = data_range

        self.x_data = list(range(0, self.x_len))
        
        # Store a reference to the figure and and axis 
        self.fig, self.ax = plt.subplots()
        self.ax.set_ylim(ymin = self.data_range[0], ymax = self.data_range[1])
        
        ## START STREAMING
        self.vw.device['reader'].start()
        self.vw.device['reader'].wait_for_reading_loop()
        sleep(5)

        # Need to determine how many channels to plot
        self.emg = self.vw.device['emg_buf'].read_matrix()
        self.emg = self.emg[:,2:18]
        self.active_channels = self.emg.sum(axis=0) != 0 # only want to plot the active channels
        self.active_channels_data = self.emg[:, self.active_channels]
        self.total_channel_num = len(self.active_channels_data[0])
        
        self.y_data = np.zeros((x_len, self.total_channel_num))

        self.line = self.ax.plot(self.x_data, self.y_data)


        anim.FuncAnimation.__init__(self, self.fig, self.updatePlot, interval = interval, blit = True)
        return

    def updatePlot(self, i):
        ## ADD DATA POINTS TO y_data
        self.y_data = np.concatenate((self.y_data, self.get_new_data()))
        #self.y_data(round(self.get_new_data(), 2)) # get new data and round it to 2 decimal places
        self.y_data = self.y_data[-self.x_len:]
        for channel in range(len(self.line)):
            self.line[channel].set_ydata(self.y_data[:,channel]) # don't know if set_ydata is still a thing
        return self.line

    def get_new_data(self):
        ## Get the data to update the line
        self.emg = self.vw.device['emg_buf'].read_matrix()
        self.emg = self.emg[:,2:18]
        self.active_channels = self.emg.sum(axis=0) != 0 # only want to plot the active channels
        self.active_channels_data = self.emg[:, self.active_channels]
        self.total_channel_num = len(self.active_channels_data[0])
        self.active_channels_data = np.array(self.active_channels_data)
        return self.active_channels_data



