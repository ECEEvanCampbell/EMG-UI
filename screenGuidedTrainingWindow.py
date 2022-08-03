from utils import PageWindow
from PyQt5 import QtWidgets
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import QThread, QObject, pyqtSignal # needed so progress bars don't stall the GUI/other processes
import time
from time import sleep
import sys
import numpy as np

class CollectionWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    def __init__(self, sgt, motion_procession, rep_procession, rest_duration, motion_duration):
        super(CollectionWorker, self).__init__()
        self.motion_procession = motion_procession
        self.rep_procession = rep_procession
        if motion_procession[0] == 'rest':
            self.timer_duration = rest_duration
        else:
            self.timer_duration = motion_duration
        self.sgt = sgt
        self.sgt.progressBar.setMaximum(self.timer_duration*10)
        self.sgt.image.setPixmap(QtGui.QPixmap("imgs/" + motion_procession[0] + '.jpg'))
        self.sgt.label.setText(motion_procession[0])
            
    def run(self):
        # whatever the duration of the timer is, there will be 10 x steps (progress bar resolution)
        count = 0

        # update the buffer matrices to have the corresponding gesture label
        # in the emg and aux data files that are output, the value in the last column will be various
        # different integers. 1000 = rest, 0 = no motion, 1 = hand open, 2 = hand closed, 
        # 3 = wrist flexion, 4 = wrist extension, 5 = pronation, 6 = supination.
        if self.motion_procession[0] == 'rest':
            class_label = 1000

        elif self.motion_procession[0] == 'no motion':
            class_label = 0
        
        elif self.motion_procession[0] == 'hand open':
            class_label = 1

        elif self.motion_procession[0] == 'hand closed':
            class_label = 2

        elif self.motion_procession[0] == 'wrist flexion':
            class_label = 3

        elif self.motion_procession[0] == 'wrist extension':
            class_label = 4

        elif self.motion_procession[0] == 'pronation':
            class_label = 5

        elif self.motion_procession[0] == 'supination':
            class_label = 6

        rep_label = self.rep_procession[0]
        new_labels = np.array([class_label, rep_label])

        self.sgt.basewindow.device['reader'].update_custom_columns(new_labels)

        while count < self.timer_duration*10:
            count += 1
            sleep(0.1)
            self.progress.emit(count)
        self.finished.emit()

    
class ScreenGuidedTrainingWindow(PageWindow):
    def __init__(self, basewindow):
        super().__init__()
        # Lets store the handle to the settings window as a 
        self.basewindow = basewindow

        # window properties specific to the collection window
        self.title_font    = self.basewindow.title_font
        self.subtitle_font = self.basewindow.subtitle_font
        self.text_font     = self.basewindow.text_font
        self.massive_font  = self.basewindow.massive_font
        
        self.setObjectName("sgt_window")

        # spawn all the widgets relevant for the collection window
        self.initUI()



        
    def onRender(self):
        if  ("name" in self.basewindow.device):
            # self.basewindow.state.append('COLLECTING')
            self.collect()
            # self.basewindow.state.remove('COLLECTING')

    def initUI(self):
        self.setWindowTitle("Screen Guided Training")
        self.UiComponents()
    
    def UiComponents(self):

        ## CENTRAL WIDGET - this is a parent widget for formatting
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.setCentralWidget(self.centralwidget)

        ## IMAGE WIDGET - this is the visual prompt of the motion the user should perform
        self.image = QtWidgets.QLabel(self.centralwidget)
        self.image.setGeometry(QtCore.QRect(30, 10, 350, 450))
        self.image.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        self.image.setText("")
        self.image.setPixmap(QtGui.QPixmap("imgs/Hand open.jpg"))
        self.image.setScaledContents(True)
        self.image.setObjectName("image")

        ## PROGRESS BAR WIDGET - this indicates how long the user should hold the gesture
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(30, 475, 511, 41))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")

        ## STOP BUTTON WIDGET - 
        self.stop_button = QtWidgets.QPushButton(self.centralwidget)
        self.stop_button.setGeometry(QtCore.QRect(570, 475, 151, 81))
        self.stop_button.setFont(self.title_font)
        self.stop_button.setObjectName("stop_button")
        
        ## MOTION LABEL WIDGET - text of prompted motion
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(150, 475, 291, 40))
        self.label.setFont(self.massive_font)
        self.label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")

        # self.retranslateUi()
        # QtCore.QMetaObject.connectSlotsByName(self)

        # ATTACH CALLBACK FUNCTIONS TO BUTTONS 
        self.stop_button.clicked.connect(self.stop_button_pressed)



    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        #self.setWindowTitle(_translate("Collection_Window", "MainWindow"))
        self.stop_button.setText(_translate("sgt_window", "Stop"))
        #self.label.setText(_translate("Collection_Window", "REST"))


    def stop_button_pressed(self):
        self.close()


    def collect(self):
        config = self.basewindow.collection_vars
        motion_procession = config['motions'] * config['reps']
        rep_procession = []

        # perform collection
        self.basewindow.device['reader'].start() # start streaming loop
        self.basewindow.device['reader'].wait_for_reading_loop()


        for r in range(config['reps']):
            if config['rest_duration'] > 0:
                factor = 2
            else:
                factor = 1
            rep_procession.extend([r]*len(config['motions'])*factor)
        if config['rest_duration'] > 0:
            motion_procession = [i for s in [[motion, 'rest'] for motion in motion_procession] for i in s]
        

        self.make_parallel_worker(self.progressBar_update, motion_procession, rep_procession, config['rest_duration'], config['motion_duration'])
        
    def make_parallel_worker(self, fun_handle, motion_procession, rep_procession, rest_duration, motion_duration):
            # if there is still some motion to progress through
            if motion_procession:
                self.thread = QThread() # define a thread (multiprocessing)
                # define a worker (functions that can be done in a single thread)
                self.worker = CollectionWorker(self, motion_procession, rep_procession, rest_duration, motion_duration)
                self.worker.moveToThread(self.thread) # assign the worker to the thread
                self.thread.started.connect(self.worker.run) # what is called when thread is started
                self.worker.finished.connect(self.thread.quit) # close thread when worker is done
                self.worker.finished.connect(self.worker.deleteLater) # when worker is done kill worker
                self.thread.finished.connect(self.thread.deleteLater) # when worker is done kill thread
                self.worker.progress.connect(fun_handle) # update progress bar according to worker
                # start the thread
                self.thread.start()
                # I never thought I'd use recursion...haha
                self.thread.finished.connect(
                    lambda: self.make_parallel_worker(fun_handle, motion_procession[1:], rep_procession[1:], rest_duration, motion_duration)
                )
            # when we are done close the collection window via the close event
            else:
                if self.basewindow.device['name'] == "Delsys":
                    self.basewindow.device['reader'].shutdown()
                    self.basewindow.device['reader'].join()
                    self.basewindow.device['emg_buf'].close()
                    self.basewindow.device['aux_buf'].close()
                self.basewindow.goto("main")
    

    def progressBar_update(self, value):
        self.progressBar.setValue(value)
