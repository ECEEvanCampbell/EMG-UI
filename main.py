'''
This file outlines the single-page GUI application that all functionality is built around.
'''

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QAction
from utils import PageWindow
from connectionWindow import ConnectionWindow
from screenGuidedTrainingWindow import ScreenGuidedTrainingWindow
from modelTrainingWindow import ModelTrainingWindow



class MainWindow(PageWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.setWindowTitle("MainWindow")

    def initUI(self):
        self.UiComponents()

    def UiComponents(self):
        self.label = QtWidgets.QLabel("EMG - UI - UNB", self)

'''
Main window class: this is the base window that all other pages are shown from.
'''
class Window(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        
        self.setGeometry(0,0,1280, 720)
        self.stacked_widget = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.m_pages = {}
        # if we want to register a new functionality (new page), include the PageWindow here
        self.register(ConnectionWindow(), "connect")
        self.register(ScreenGuidedTrainingWindow(), "screenguidedtraining")
        self.register(ModelTrainingWindow(), "modeltraining")
        
        self.register(MainWindow(), "main")
        

        self.goto("main")

        self.setup_menubar()
    
    def setup_menubar(self):
        menubar = self.menuBar()
        filemenu = menubar.addMenu("File")
        connection_action = filemenu.addAction("Connect")
        connection_action.setStatusTip("Configure and connect to instrumentation hardware.")
        connection_action.triggered.connect(lambda: self.goto("connect"))

        collectmenu = menubar.addMenu("Collect")
        collect_action = collectmenu.addAction("Screen Guided Training")
        collect_action.setStatusTip("Collect signals while prompting the user using images.")
        collect_action.triggered.connect(lambda: self.goto("screenguidedtraining"))

        trainmenu = menubar.addMenu("Train")
        train_action = trainmenu.addAction("Prepare ML Model")
        train_action.setStatusTip("Prepare a pipeline for EMG gesture recognition using collected data.")
        train_action.triggered.connect(lambda: self.goto("modeltraining"))

    def register(self, widget, name):
        self.m_pages[name] = widget
        self.stacked_widget.addWidget(widget)
        if isinstance(widget, PageWindow):
            widget.gotoSignal.connect(self.goto)

    @QtCore.pyqtSlot(str)
    def goto(self, name):
        if name in self.m_pages:
            widget = self.m_pages[name]
            self.stacked_widget.setCurrentWidget(widget)
            self.setWindowTitle(widget.windowTitle())


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())