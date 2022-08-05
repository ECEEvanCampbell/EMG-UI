from time import sleep
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QWidget
import sys
import matplotlib

from utils import PageWindow
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
import queue
import numpy as np
import pdb
from PyQt5 import QtCore, QtWidgets,QtGui
from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot

class myCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        myfig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = myfig.add_subplot(111)
        super(myCanvas, self).__init__(myfig)
        myfig.tight_layout()

class visWindow(PageWindow):
    def __init__(self, basewindow):
        super(visWindow, self).__init__()
        self.resize(1153, 747)

        self.basewindow = basewindow
        self.setupUi()
    
    def setupUi(self):
        # ## MAIN WINDOW PARAMETERS
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.setCentralWidget(self.centralwidget)

        # # MENU/STATUS BAR
        # self.menubar = QtWidgets.QMenuBar(self)
        # self.menubar.setGeometry(QtCore.QRect(0, 0, 1153, 26))
        # self.menubar.setObjectName("menubar")
        # self.setMenuBar(self.menubar)
        # self.statusbar = QtWidgets.QStatusBar(self)
        # self.statusbar.setObjectName("statusbar")
        # self.setStatusBar(self.statusbar)
        
        ## VERTICAL LAYOUT
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(390, 10, 731, 641))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")

        ## FIGURE PARAMETERS
        self.plotWindow = QtWidgets.QWidget(self.verticalLayoutWidget)
        self.plotWindow.setGeometry(QtCore.QRect(390, 10, 731, 641))
        self.plotWindow.setObjectName("plotWindow")
        self.verticalLayout.addWidget(self.plotWindow)
        
        ## INTERVAL INPUT SETUP
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 30, 171, 51))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.updateInterval = QtWidgets.QLineEdit(self.centralwidget)
        self.updateInterval.setGeometry(QtCore.QRect(200, 45, 113, 22))
        self.updateInterval.setObjectName("updateInterval")
        self.updateInterval.setText('100') # set the update interval to 0.1s by default

        ## START BUTTON
        self.title_font = QtGui.QFont()
        self.title_font.setPointSize(16)


        ## POPULATE FIELDS WITH TEXT
        #self.retranslateUi(self)
        #QtCore.QMetaObject.connectSlotsByName(self)

        ## CONNECT CALLBACKS IF UPDATE INTERVAL CHANGES
        self.updateInterval.textChanged['QString'].connect(self.updateIntervalChanged)



        ## SET UP CANVAS + TIMER
        self.threadpool = QtCore.QThreadPool()
        self.figureCanvas = myCanvas(self, width=5, height=4, dpi=100)
        self.verticalLayout.addWidget(self.figureCanvas, 2)
        self.plot_reference = None

        #self.data = np.zeros() # input dimensions here

        #self.updatePlot()
        self.timer = QtCore.QTimer()
        self.timer.setInterval(int(self.updateInterval.text()))
        self.timer.timeout.connect(self.updatePlot)


        self.start_button = QtWidgets.QPushButton(self.centralwidget)
        self.start_button.setGeometry(QtCore.QRect(20, 100, 200, 40))
        self.start_button.setFont(self.title_font)
        self.start_button.clicked.connect(self.startPressed)
        self.start_button.setText("Start")

    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "Visualization Window"))
        self.label.setText(_translate("mainWindow", "Update Interval (ms)"))

    def updateIntervalChanged(self):
        self.timer.setInterval(int(self.updateInterval.text()))
        self.timer.timeout.connect(self.updatePlot)
        #self.timer.start()

    def startPressed(self):
        ## START STREAMING DATA
        self.basewindow.device['reader'].start()
        self.basewindow.device['reader'].wait_for_reading_loop()
        sleep(2)
        
        worker = Worker(self.getEMG, )
        self.threadpool.start(worker)

        # # ## START THE TIMER
        self.timer.start()

        
    def getEMG(self):
        ## RESTRICT CHANGE IN INTERVAL UPDATE FIELD
        self.updateInterval.setEnabled(False)
        while True:
            self.emg = self.basewindow.device['emg_buf'].read_matrix()
            self.emg = self.emg[:,2:18]
            self.active_channels = self.emg.sum(axis=0) != 0 # only want to plot the active channels
            self.active_channels_data = self.emg[:, self.active_channels]
            self.total_channel_num = len(self.active_channels_data[0])
            sleep(0.01)

        
        

    def updatePlot(self):
        self.y_data = self.active_channels_data

        if self.plot_reference is None:
            ref = self.figureCanvas.ax.plot(self.y_data)
            self.plot_reference = ref
        else:
            for data, line in zip(self.y_data.T, self.plot_reference):
                line.set_data(list(range(data.shape[0])), data)
        self.figureCanvas.draw()
        #print(self.active_channels_data.shape)
    
    def closeEvent(self, event):
        ## CLOSE THE BUFFERS
        self.basewindow.device['reader'].shutdown()
        self.basewindow.device['reader'].join()
        self.basewindow.device['emg_buf'].close()
        self.basewindow.device['aux_buf'].close()

        ## NOW CLOSE THE WINDOW
        event.accept()

class Worker(QtCore.QRunnable):
    def __init__(self, function, *args, **kwargs):
        super(Worker, self).__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs

    @pyqtSlot()
    def run(self):
        self.function(*self.args, **self.kwargs)
        


# if __name__ == "__main__":
#     import sys
#     app = QtWidgets.QApplication(sys.argv)
#     mainWindow = QtWidgets.QMainWindow()
#     ui = Ui_mainWindow()
#     #ui.setupUi(mainWindow)
#     mainWindow.show()
#     sys.exit(app.exec_())
