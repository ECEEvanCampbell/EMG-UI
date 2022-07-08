from utils import PageWindow
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QComboBox, QMessageBox

class ScreenGuidedTrainingSetupWindow(PageWindow):
    def __init__(self, basewindow):
        super().__init__()

        self.basewindow = basewindow

        self.setObjectName("settings_window")
        self.massive_font = QtGui.QFont()
        self.massive_font.setPointSize(22)
        self.title_font = QtGui.QFont()
        self.title_font.setPointSize(16)
        self.subtitle_font = QtGui.QFont()
        self.subtitle_font.setPointSize(14)
        self.text_font  = QtGui.QFont()
        self.text_font.setPointSize(11)
        self.initUI()


        self.gestures_selected = []
        self.rep_num_value = 0
        self.rep_duration_value = 0
        self.rest_duration_value = 0

    def initUI(self):
        self.setWindowTitle("Screen Guided Training Setup")
        self.UiComponents()

    def UiComponents(self):
        ## CENTRAL WIDGET - this is a parent widget for formatting
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.setCentralWidget(self.centralwidget)

        ## GESTURE LIST WIDGET
        self.gesture_list_label = QtWidgets.QLabel(self.centralwidget)
        self.gesture_list_label.setGeometry(QtCore.QRect(40, 10, 131, 31))
        self.gesture_list_label.setFont(self.subtitle_font)
        self.gesture_list_label.setObjectName("gesture_list_label")

        ## CENTRAL WIDGET - FRAME FOR COMBO
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(20, 50, 221, 311))
        self.frame.setFont(self.text_font)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")

        ## CHECKBOXES START
        # No MOTION
        self.no_motion = QtWidgets.QCheckBox(self.frame)
        self.no_motion.setGeometry(QtCore.QRect(20, 20, 81, 20))
        self.no_motion.setObjectName("no_motion")
        # HAND OPEN MOTION
        self.hand_open = QtWidgets.QCheckBox(self.frame)
        self.hand_open.setGeometry(QtCore.QRect(20, 60, 121, 20))
        self.hand_open.setObjectName("hand_open")
        # HAND CLOSED MOTION
        self.hand_closed = QtWidgets.QCheckBox(self.frame)
        self.hand_closed.setGeometry(QtCore.QRect(20, 100, 131, 20))
        self.hand_closed.setObjectName("hand_closed")
        # WRIST FLEXION MOTION
        self.wrist_flexion = QtWidgets.QCheckBox(self.frame)
        self.wrist_flexion.setGeometry(QtCore.QRect(20, 140, 131, 20))
        self.wrist_flexion.setObjectName("wrist_flexion")
        # WRIST EXTENSION MOTION
        self.wrist_extension = QtWidgets.QCheckBox(self.frame)
        self.wrist_extension.setGeometry(QtCore.QRect(20, 180, 151, 20))
        self.wrist_extension.setObjectName("wrist_extension")
        # WRIST PRONATION MOTION
        self.pronation = QtWidgets.QCheckBox(self.frame)
        self.pronation.setGeometry(QtCore.QRect(20, 220, 151, 20))
        self.pronation.setObjectName("pronation")
        # WRIST SUPINATION MOTION
        self.supination = QtWidgets.QCheckBox(self.frame)
        self.supination.setGeometry(QtCore.QRect(20, 260, 151, 20))
        self.supination.setObjectName("supination") 

        ## CENTRAL WIDGET - FRAME FOR COLLECTION PARAMETERS
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(270, 40, 221, 150))
        self.frame_2.setFont(self.text_font)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")

        ## COLLECTION PARAMTER 
        self.rep_duration_label = QtWidgets.QLabel(self.frame_2)
        self.rep_duration_label.setGeometry(QtCore.QRect(80, 30, 111, 41))
        self.rep_duration_label.setObjectName("rep_duration_label")
        self.rep_num = QtWidgets.QSpinBox(self.frame_2)
        self.rep_num.setGeometry(QtCore.QRect(20, 0, 42, 22))
        self.rep_num.setObjectName("rep_num")
        self.rep_duration = QtWidgets.QSpinBox(self.frame_2)
        self.rep_duration.setGeometry(QtCore.QRect(20, 40, 42, 22))
        self.rep_duration.setObjectName("rep_duration")
        self.rep_num_label = QtWidgets.QLabel(self.frame_2)
        self.rep_num_label.setGeometry(QtCore.QRect(80, -10, 141, 41))
        self.rep_num_label.setFont(self.text_font)
        self.rep_num_label.setObjectName("rep_num_label")
        self.rest_duration = QtWidgets.QSpinBox(self.frame_2)
        self.rest_duration.setGeometry(QtCore.QRect(20, 80, 42, 22))
        self.rest_duration.setObjectName("rest_duration")
        self.rest_duration_label = QtWidgets.QLabel(self.frame_2)
        self.rest_duration_label.setGeometry(QtCore.QRect(80, 70, 111, 41))
        self.rest_duration_label.setObjectName("rest_duration_label")

        ## SPINBOX TO PICK SENSOR NUMBER TO DISPLAY
        self.sensor_num = QtWidgets.QSpinBox(self.frame_2)
        self.sensor_num.setGeometry(QtCore.QRect(20, 120, 42, 22))
        self.sensor_num.setObjectName("sensor_num")
        self.sensor_num.setValue(1)
        self.sensor_num.hide()
        self.sensor_num_label = QtWidgets.QLabel(self.frame_2)
        self.sensor_num_label.setGeometry(QtCore.QRect(80, 110, 111, 41))
        self.sensor_num_label.setObjectName("sensor_num_label")
        self.sensor_num_label.hide()

        ## BUTTONS
        # # CONENCT BUTTON
        # self.connect_button = QtWidgets.QPushButton(self.centralwidget)
        # self.connect_button.setGeometry(QtCore.QRect(290, 280, 131, 41))
        # self.connect_button.setFont(self.title_font)
        # self.connect_button.setObjectName("connect_button")
        # START BUTTON
        self.start_button = QtWidgets.QPushButton(self.centralwidget)
        self.start_button.setGeometry(QtCore.QRect(290, 330, 131, 41))
        self.start_button.setFont(self.title_font)
        self.start_button.setObjectName("start_button")
        # VISUALIZATION WINDOW
        self.visualize_button = QtWidgets.QPushButton(self.centralwidget)
        self.visualize_button.setGeometry(QtCore.QRect(290, 380, 131, 41))
        self.visualize_button.setFont(self.title_font)
        self.visualize_button.setObjectName("visualization_button")

        ## ACTUALLY POPULATE VISABLE LABEL FIELDS W/ TEXT
        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

        ## ATTACH CALLBACK FUNCTIONS TO BUTTONS / CHECKBOXES
        self.start_button.clicked.connect(self.start_pressed)
        
        # self.visualize_button.clicked.connect(self.visualize_pressed)
        self.no_motion.stateChanged.connect(self.no_motion_gesture_clicked)
        self.hand_open.stateChanged.connect(self.hand_open_gesture_clicked)
        self.hand_closed.stateChanged.connect(self.hand_closed_gesture_clicked)
        self.wrist_flexion.stateChanged.connect(self.wrist_flexion_gesture_clicked)
        self.wrist_extension.stateChanged.connect(self.wrist_extension_gesture_clicked)
        self.pronation.stateChanged.connect(self.pronation_gesture_clicked)
        self.supination.stateChanged.connect(self.supination_gesture_clicked)
        self.rep_num.valueChanged.connect(self.rep_num_changed)
        self.rep_duration.valueChanged.connect(self.rep_duration_changed)
        self.rest_duration.valueChanged.connect(self.rest_duration_changed)
        # self.combobox.activated[str].connect(self.system_selected)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("settings_window", "MainWindow"))
        self.gesture_list_label.setText(_translate("settings_window", "Gesture List"))
        self.no_motion.setText(_translate("settings_window", "No Motion"))
        self.hand_open.setText(_translate("settings_window", "Hand open"))
        self.hand_closed.setText(_translate("settings_window", "Hand closed"))
        self.wrist_flexion.setText(_translate("settings_window", "Wrist flexion"))
        self.wrist_extension.setText(_translate("settings_window", "Wrist extension"))
        self.pronation.setText(_translate("settings_window", "Pronation"))
        self.supination.setText(_translate("settings_window", "Supination"))
        self.rep_duration_label.setText(_translate("settings_window", "Rep duration"))
        self.rep_num_label.setText(_translate("settings_window", "Number of reps"))
        self.rest_duration_label.setText(_translate("settings_window", "Rest duration"))
        self.sensor_num_label.setText(_translate("settings_window", "Sensor Num"))
        # self.connect_button.setText(_translate("settings_window", "Connect"))
        self.start_button.setText(_translate("settings_window", "Start"))
        # self.combo_box_label.setText(_translate("settings_window", "System Selection:"))
        self.visualize_button.setText(_translate("settings_window", "Visualize"))

    def no_motion_gesture_clicked(self, state):
        if (QtCore.Qt.Checked == state):
            self.gestures_selected.append("no motion") 
        else:
            self.gestures_selected.remove("no motion")

    def hand_open_gesture_clicked(self, state):
        if (QtCore.Qt.Checked == state):
            self.gestures_selected.append("hand open")    
        else:
            self.gestures_selected.remove("hand open")

    def hand_closed_gesture_clicked(self, state):
        if (QtCore.Qt.Checked == state):
            self.gestures_selected.append("hand closed")    
        else:
            self.gestures_selected.remove("hand closed")
    
    def wrist_flexion_gesture_clicked(self, state):
        if (QtCore.Qt.Checked == state):
            self.gestures_selected.append("wrist flexion")    
        else:
            self.gestures_selected.remove("wrist flexion")
    
    def wrist_extension_gesture_clicked(self, state):
        if (QtCore.Qt.Checked == state):
            self.gestures_selected.append("wrist extension")    
        else:
            self.gestures_selected.remove("wrist extension")

    def pronation_gesture_clicked(self, state):
        if (QtCore.Qt.Checked == state):
            self.gestures_selected.append("pronation")    
        else:
            self.gestures_selected.remove("pronation")

    def supination_gesture_clicked(self, state):
        if (QtCore.Qt.Checked == state):
            self.gestures_selected.append("supination")    
        else:
            self.gestures_selected.remove("supination")

    def rep_num_changed(self):
        self.rep_num_value = self.rep_num.value()

    def rep_duration_changed(self):
        self.rep_duration_value = self.rep_duration.value()

    def rest_duration_changed(self):
        self.rest_duration_value = self.rest_duration.value()

    def system_selected(self):
        self.sensor = self.combobox.currentText()    

    

    def start_pressed(self):
        # if the sensor isn't connected we shouldn't proceed to collecting data
        if not ("name" in self.basewindow.device) :
            msg = QMessageBox()
            msg.setWindowTitle("Error")
            msg.setText("No sensor is connected")
            msg.setIcon(QMessageBox.Critical)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return
        
        # get all the info from the current settings
        self.basewindow.collection_vars = self.get_settings()
        self.basewindow.goto('screenguidedtraining')
        
        

    def get_settings(self):
        config = {
            'motions': self.gestures_selected,
            'reps'   : self.rep_num_value,
            'motion_duration': self.rep_duration_value,
            'rest_duration'  : self.rest_duration_value
        }
        return config