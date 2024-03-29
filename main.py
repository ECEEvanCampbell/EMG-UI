'''
This file outlines the single-page GUI application that all functionality is built around.
'''

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QAction
from utils import PageWindow
from connectionWindow import ConnectionWindow
from screenGuidedTrainingSetupWindow import ScreenGuidedTrainingSetupWindow
from screenGuidedTrainingWindow import ScreenGuidedTrainingWindow
#from modelTrainingWindow import ModelTrainingWindow
from modelManualTrainingWindow import ManualModelTrainingWindow
from modelSelectTrainingWindow import SelectModelTrainingWindow
from modelManageWindow import ManageModelTrainingWindow
from fittsLawSetupWindow import FittsLawSetupWindow
#from visualizationWindowSetup import VisualizationWindowSetup
from visualizationWindow import VisualizationWindow
from livePlotWindow import visWindow


# think of this as a greeting screen or something
class MainWindow(PageWindow):
    def __init__(self, basewindow):
        super().__init__()
        self.basewindow = basewindow
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


        self.massive_font = QtGui.QFont()
        self.massive_font.setPointSize(22)
        self.title_font = QtGui.QFont()
        self.title_font.setPointSize(16)
        self.subtitle_font = QtGui.QFont()
        self.subtitle_font.setPointSize(14)
        self.text_font  = QtGui.QFont()
        self.text_font.setPointSize(11)
        
        self.state = []
        self.device = {}
        self.model  = {}
        
        self.m_pages = {}
        # if we want to register a new functionality (new page), include the PageWindow here
        self.register(ConnectionWindow(self), "connect")
        self.register(ScreenGuidedTrainingSetupWindow(self), "screenguidedtrainingsetup")
        self.register(ScreenGuidedTrainingWindow(self), "screenguidedtraining")
        self.register(ManualModelTrainingWindow(self), "manualmodeltraining")
        self.register(SelectModelTrainingWindow(self), "selectmodeltraining")
        self.register(ManageModelTrainingWindow(self), "managemodeltraining")
        self.register(FittsLawSetupWindow(self), "fittslawsetup")
        #self.register(VisualizationWindowSetup(self), "visualizationsetup")
        #self.register(VisualizationWindow(self), "visualization")
        self.register(visWindow(self), "visualization")
        
        self.register(MainWindow(self), "main")
        
        

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
        collect_action.triggered.connect(lambda: self.goto("screenguidedtrainingsetup"))
        collect_action2 = collectmenu.addAction("Visualize")
        collect_action2.setStatusTip("View signals while collecting data")
        #self.vis_window = visWindow(self)
        #collect_action2.triggered.connect(lambda: #self.vis_window.setupUi())
        collect_action2.triggered.connect(lambda: self.goto("visualization"))

        trainmenu = menubar.addMenu("Train")
        
        manual_train_action = trainmenu.addAction("Manual ML Model")
        manual_train_action.setStatusTip("Prepare a pipeline for EMG gesture recognition using collected data.")
        manual_train_action.triggered.connect(lambda: self.goto("manualmodeltraining"))
        
        select_train_action = trainmenu.addAction("Selection Based ML Model")
        select_train_action.setStatusTip("Determine best features for model from collected data.")
        select_train_action.triggered.connect(lambda: self.goto("selectmodeltraining"))

        manage_train_action = trainmenu.addAction("Manage ML Model")
        manage_train_action.setStatusTip("Load/Save Models.")
        manage_train_action.triggered.connect(lambda: self.goto("managemodeltraining"))

        testmenu = menubar.addMenu("Test")
        test_action = testmenu.addAction("Fitts Law Test")
        test_action.setStatusTip("Use a trained model to perform an online evaluation (Fitts Law).")
        test_action.triggered.connect(lambda: self.goto("fittslawsetup"))

    def register(self, widget, name):
        self.m_pages[name] = widget
        self.stacked_widget.addWidget(widget)
        if isinstance(widget, PageWindow):
            widget.gotoSignal.connect(self.goto)

    @QtCore.pyqtSlot(str)
    def goto(self, name):
        if name in self.m_pages:
            widget = self.m_pages[name]
            widget.onRender()
            self.stacked_widget.setCurrentWidget(widget)
            self.setWindowTitle(widget.windowTitle())


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())