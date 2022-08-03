from utils import PageWindow
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QComboBox, QMessageBox, QLineEdit, QLabel
from biomedical_hardware_readers import DelsysTrignoReader
import numpy as np

class ConnectionWindow(PageWindow):
    def __init__(self, basewindow):
        super().__init__()
        self.basewindow = basewindow
        self.massive_font = QtGui.QFont()
        self.massive_font.setPointSize(22)
        self.title_font = QtGui.QFont()
        self.title_font.setPointSize(16)
        self.subtitle_font = QtGui.QFont()
        self.subtitle_font.setPointSize(14)
        self.text_font  = QtGui.QFont()
        self.text_font.setPointSize(11)
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Device Connection")
        self.UiComponents()

    def goToMain(self):
        self.goto("main")

    def UiComponents(self):
        ## CENTRAL WIDGET - this is a parent widget for formatting
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.setCentralWidget(self.centralwidget)

        ## filename Label
        self.filename_label = QLabel(self.centralwidget)
        self.filename_label.setText("Filename:")
        self.filename_label.setFont(self.subtitle_font)
        self.filename_label.setGeometry(QtCore.QRect(20, 10, 200, 40))
        ## filename area
        self.filename_input = QLineEdit("",self.centralwidget)
        self.filename_input.setGeometry(QtCore.QRect(20, 60, 200, 40))

        ## Windowsize Label
        self.buffer_duration_label = QLabel(self.centralwidget)
        self.buffer_duration_label.setText("Buffer Duration:")
        self.buffer_duration_label.setFont(self.subtitle_font)
        self.buffer_duration_label.setGeometry(QtCore.QRect(20, 110, 200, 40))
        ## Windowsize area
        self.buffer_duration_input = QLineEdit(".250",self.centralwidget)
        self.buffer_duration_input.setGeometry(QtCore.QRect(20, 160, 200, 40))


        ## COMBO BOX
        self.combobox = QComboBox(self.centralwidget)
        self.combobox.addItems(['Select', 'Delsys', 'SiFi']) 
        self.combobox.setGeometry(QtCore.QRect(20, 260, 200, 40))
        self.combobox.activated[str].connect(self.system_selected)

        self.combo_box_label = QtWidgets.QLabel(self.centralwidget)
        self.combo_box_label.setGeometry(QtCore.QRect(20, 210, 200, 40))
        self.combo_box_label.setFont(self.text_font)
        self.combo_box_label.setText("Select Device Type:")

        # CONNECT BUTTON
        self.connect_button = QtWidgets.QPushButton(self.centralwidget)
        self.connect_button.setGeometry(QtCore.QRect(20, 310, 200, 40))
        self.connect_button.setFont(self.title_font)
        self.connect_button.clicked.connect(self.connect_pressed)
        self.connect_button.setText("Connect")

        # DISCONNECT BUTTON
        self.disconnect_button = QtWidgets.QPushButton(self.centralwidget)
        self.disconnect_button.setGeometry(QtCore.QRect(20, 360, 200, 40))
        self.disconnect_button.setFont(self.title_font)
        self.disconnect_button.clicked.connect(self.disconnect_pressed)
        self.disconnect_button.setText("Disconnect")



    def system_selected(self):
        self.sensor = self.combobox.currentText()


    def connect_pressed(self):
        # add code to connect to delsys system
        try:
            if self.sensor == "Delsys":
                #do delsys stuff
                filename = self.filename_input.text()
                # TODO: add check to see if file exists (if exists throw error and prompt user for new name.)
                self.basewindow.device['reader'] = DelsysTrignoReader(file_exist_ok=True, emg_file_name=filename+'_EMG.csv', aux_file_name=filename+'_AUX.csv')
                self.basewindow.device['reader'].register_custom_columns(num_columns=2)
                rnd_number = np.random.randint(1e6)
                buffer_length = float(self.buffer_duration_input.text())
                (self.basewindow.device['emg_buf'], self.basewindow.device['aux_buf']) = self.basewindow.device['reader'].create_shared_matrix(buffer_duration=buffer_length, emg_shared_matrix_name="Electromyography"+str(rnd_number), aux_shared_matrix_name='Auxilliary'+str(rnd_number))
                self.basewindow.device['name'] = 'Delsys'
            elif self.sensor == "SiFi":
                # do SiFi stuff
                self.basewindow.device['name'] = 'SiFi'
                pass

            msg = QMessageBox()
            msg.setWindowTitle("Sensor Status")
            msg.setText("Sensor is connected")
            msg.setIcon(QMessageBox.Information)
            msg.setStandardButtons(QMessageBox.Ok)
            x = msg.exec_() # shows the message box
            self.basewindow.state.append('CONNECTED')
            self.basewindow.goto("main")

        
        except:
            msg = QMessageBox()
            msg.setWindowTitle("Sensor Status")
            msg.setText("Sensor connection failed")
            msg.setIcon(QMessageBox.Critical)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_() # shows the message box

            self.basewindow.device = {}

    def disconnect_pressed(self):
        try:
            if self.sensor == "Delsys":
                if not self.basewindow.device['reader']._closed:
                    self.basewindow.device['reader'].shutdown()
                    self.basewindow.device['reader'].join()
                    self.basewindow.device['emg_buf'].close()
                    self.basewindow.device['aux_buf'].close()
                    self.basewindow.state.remove('CONNECTED')
            elif self.sensor == "SiFi":
                pass
        except:
            msg = QMessageBox()
            msg.setWindowTitle("Sensor Status")
            msg.setText("Sensor disconnection failed")
            msg.setIcon(QMessageBox.Critical)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_() # shows the message box