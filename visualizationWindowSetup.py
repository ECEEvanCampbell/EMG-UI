from utils import PageWindow
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QMessageBox

class VisualizationWindowSetup(PageWindow):
    def __init__(self, basewindow):
        super().__init__()

        self.basewindow = basewindow

        self.setObjectName("visualization_window")
        self.text_font = QtGui.QFont()
        self.text_font.setPointSize(11)
        self.title_font = QtGui.QFont()
        self.title_font.setPointSize(16)
        self.initUI()

        self.sensor_num_value = 1 # channel 1 by default will be selected

    def initUI(self):
        self.setWindowTitle("Visualization Setup")
        self.UiComponents()

    def UiComponents(self):
        ## CENTRAL WIDGET
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.setCentralWidget(self.centralwidget)

        ## FRAME FOR PARAMETERS
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(0, 0, 200, 300)) # modify this as needed
        self.frame.setFont(self.text_font)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")

        ## CHANNEL SELECTION - SPINBOX
        self.sensor_num = QtWidgets.QSpinBox(self.frame)
        self.sensor_num.setGeometry(QtCore.QRect(10, 10, 42, 22))
        self.sensor_num.setObjectName("sensor_num")
        self.sensor_num.setValue(1)
        self.sensor_num_label = QtWidgets.QLabel(self.frame)
        self.sensor_num_label.setGeometry(QtCore.QRect(60, 10, 120, 20))
        self.sensor_num_label.setObjectName("sensor_num_label")

        ## VISUALIZE BUTTON
        self.visualize_button = QtWidgets.QPushButton(self.centralwidget)
        self.visualize_button.setGeometry(QtCore.QRect(10, 50, 131, 41))
        self.visualize_button.setFont(self.title_font)
        self.visualize_button.setObjectName("visualize_button")
        
        ## POPULATE VISIBLE LABELS WITH TEXT
        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

        ## ATTACH CALLBACK FUNCTIONS TO BUTTONS/SPINBOXES
        self.sensor_num.valueChanged.connect(self.sensor_num_changed)
        self.visualize_button.clicked.connect(self.visualize_pressed)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.sensor_num_label.setText(_translate("visualization_window", "Sensor Number"))
        self.visualize_button.setText(_translate("visualization_window", "Visualize"))

    def sensor_num_changed(self):
        self.sensor_num_value = self.sensor_num.value()

    def visualize_pressed(self):
        # if the sensor isn't connected we shouldn't proceed to collecting data
        if not ("name" in self.basewindow.device):
            msg = QMessageBox()
            msg.setWindowTitle("Error")
            msg.setText("No sensor is connected")
            msg.setIcon(QMessageBox.Critical)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return
        
        # get all the info from the current settings
        self.basewindow.vis_vars = self.get_settings() # if more parameters are added
        self.basewindow.goto('visualization')

    def get_settings(self):
        config = {
            'sensor num': self.sensor_num_value
        }
        return config
    

        
