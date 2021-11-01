from fbs_runtime.application_context.PyQt5 import ApplicationContext
from PyQt5.QtWidgets import (QWidget, QLabel, QPushButton, QFileDialog,
                             QGridLayout, QComboBox, QMessageBox, QGroupBox,
                             QVBoxLayout, QSlider, QProgressDialog)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap
import sys
import os
from pandas import read_csv
from model0 import fit_model_0
from model1 import fit_model_1


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Covid-19 Rapid Mortality Surveillance')
        openva_pixmap = QPixmap('src/main/icons/openva-logo.png')
        self.label_openva = QLabel()
        self.label_openva.setPixmap(openva_pixmap.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        vitalstrat_pixmap = QPixmap('src/main/icons/Vital-Strategies-Logo.png')
        self.label_vitalstrat = QLabel()
        self.label_vitalstrat.setPixmap(vitalstrat_pixmap.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        self.resize(500, 500)

        self.btn_load_train_data = QPushButton('Load training data (.csv)')
        self.btn_load_train_data.clicked.connect(self.load_train_data)
        self.label_train_data = QLabel('(no data loaded)')
        self.train_data = None
        self.train_data_loaded = False

        self.btn_load_test_data = QPushButton('Load testing data (.csv)')
        self.btn_load_test_data.clicked.connect(self.load_test_data)
        self.label_test_data = QLabel('(no data loaded)')
        self.test_data = None
        self.test_data_loaded = False

        self.label_set_model = QLabel('Choose model:')
        self.btn_set_model = QComboBox()
        self.btn_set_model.addItems(('model 0', 'model 1'))
        self.btn_set_model.activated[str].connect(self.set_model)
        self.model_choice = "model 0"
        self.model0_classifier = "ADA"
        self.btn_model0_classifier = None
        self.groupbox_model0 = self.create_model0_groupbox()
        self.model1_iterations = 2000
        self.model1_slider = QSlider()
        self.groupbox_model1 = QGroupBox('Model 1: Options')
        self.model1_slider = QSlider(Qt.Horizontal, self.groupbox_model1)
        self.label_model1_iterations = QLabel('value: 2000')
        self.setup_model1_groupbox()

        self.btn_run_model = QPushButton('Run model')
        self.btn_run_model.clicked.connect(self.run_model)
        self.label_model_output = QLabel('(no results)')
        self.model_results = None
        self.btn_download_results = QPushButton('Download results')
        self.btn_download_results.clicked.connect(self.download_results)

        self.btn_exit = QPushButton('Exit')
        self.btn_exit.clicked.connect(self.close)

        self.grid = QGridLayout(self)
        self.grid.addWidget(self.btn_load_train_data, 0, 0)
        self.grid.addWidget(self.label_train_data, 0, 1)
        self.grid.addWidget(self.btn_load_test_data, 1, 0)
        self.grid.addWidget(self.label_test_data, 1, 1)
        self.grid.addWidget(self.label_set_model, 2, 0)
        self.grid.addWidget(self.btn_set_model, 3, 0, alignment=Qt.AlignTop)
        self.grid.addWidget(self.groupbox_model0, 4, 0, 1, 2)
        self.grid.addWidget(self.btn_run_model, 5, 0)
        self.grid.addWidget(self.label_model_output, 5, 1, 1, 2)
        self.grid.addWidget(self.btn_download_results, 6, 0)
        self.grid.addWidget(self.label_openva, 7, 0)
        self.grid.addWidget(self.label_vitalstrat, 7, 1)
        self.grid.addWidget(self.btn_exit, 7, 2)

    def load_train_data(self):
        path = QFileDialog.getOpenFileName(self, 'Open a CSV file', '', 'All Files(*.*)')
        if path != ('', ''):
            self.train_data = read_csv(path[0])
            n_train = self.train_data.shape[0]
            self.label_train_data.setText(f'Training data loaded: {n_train} deaths')
            self.train_data_loaded = True

    def load_test_data(self):
        path = QFileDialog.getOpenFileName(self, 'Open a CSV file', '', 'All Files(*.*)')
        if path != ('', ''):
            self.test_data = read_csv(path[0])
            n_test = self.test_data.shape[0]
            self.label_test_data.setText(f'Testing data loaded: {n_test} deaths')
            self.test_data_loaded = True

    def set_model(self):
        self.model_choice = self.btn_set_model.currentText()
        if self.model_choice == 'model 0':
            self.grid.removeWidget(self.groupbox_model1)
            self.groupbox_model1.deleteLater()
            self.groupbox_model1 = None
            self.groupbox_model0 = self.create_model0_groupbox()
            self.grid.addWidget(self.groupbox_model0, 4, 0, 1, 2)
        else:
            self.grid.removeWidget(self.groupbox_model0)
            self.groupbox_model0.deleteLater()
            self.groupbox_model0 = None
            self.groupbox_model1 = QGroupBox('Model 1: Options')
            self.model1_slider = QSlider(Qt.Horizontal, self.groupbox_model1)
            self.label_model1_iterations = QLabel('value: 2000')
            self.setup_model1_groupbox()
            self.grid.addWidget(self.groupbox_model1, 4, 0, 1, 2)

    def set_model0_classifier(self):
        self.model0_classifier = self.btn_set_model0_classifier.currentText()

    def set_model1_iterations(self):
        self.model1_iterations = self.model1_slider.value()
        self.label_model1_iterations.setText(f'value: {self.model1_iterations}')

    def create_model0_groupbox(self):
        group_box = QGroupBox('Model 0: Options')
        font = QFont()
        font.setPointSize(13)
        group_box.setFont(font)
        btn_classifier = QComboBox()
        btn_classifier.addItems(('ADA', 'LR'))
        layout = QVBoxLayout()
        layout.addWidget(btn_classifier)
        layout.addStretch(1)
        group_box.setLayout(layout)
        return group_box

    def setup_model1_groupbox(self):
        label = QLabel('Number of iterations')
        self.model1_slider.setRange(1000, 6000)
        self.model1_slider.setValue(2000)
        self.model1_slider.valueChanged.connect(self.set_model1_iterations)
        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.model1_slider)
        layout.addWidget(self.label_model1_iterations)
        layout.addStretch(1)
        self.groupbox_model1.setLayout(layout)

    def run_model(self):
        if (not self.test_data_loaded) or (not self.train_data_loaded):
            alert = QMessageBox()
            alert.setText('Need training and/or testing data. Please click ' +
                          '"Load training/testing data (.csv)" buttons to select data file.')
            alert.exec()
        else:
            if self.model_choice == 'model 0':
                self.model_results = fit_model_0(train_data=self.train_data,
                                                 test_data=self.test_data,
                                                 id_column_name='ID',
                                                 covid_column_name='covid',
                                                 classifier=self.model0_classifier)
                results_mean = round(self.model_results["prevalence"], 3)
                self.label_model_output.setText(f'Predicted prevalence: {results_mean}')
            else:
                self.progress_bar = QProgressDialog('Running Model 1...', 'Cancel', 0, 100, self)
                self.progress_bar.setWindowModality(Qt.WindowModal)
                burn_in = int(self.model1_iterations/2)
                self.model_results = fit_model_1(train_data=self.train_data,
                                                 test_data=self.test_data,
                                                 id_column_name='ID',
                                                 covid_column_name='covid',
                                                 app_instance=self,
                                                 n_iter=self.model1_iterations,
                                                 burn_in=burn_in,
                                                 pooled=True)
                results_mean = round(self.model_results['prevalence'].mean()[0], 3)
                results_ci95 = self.model_results['prevalence'].quantile(q=[.025, .975])
                results_p025 = round(results_ci95.iloc[0], 3)[0]
                results_p975 = round(results_ci95.iloc[1], 3)[0]
                self.label_model_output.setText(f'Predicted prevalence (95% CI): {results_mean} '
                                                f'({results_p025}, {results_p975})')

    def show_plots(self):  # disable until results are ready
        if not self.data_loaded:
            alert = QMessageBox()
            alert.setText('No data loaded. Please click "Load data (.csv)" button to select data file.')
            alert.exec()
        else:
            pass

    def download_plots(self):
        if not self.data_loaded:
            alert = QMessageBox()
            alert.setText('No data loaded. Please click "Load data (.csv)" button to select data file.')
            alert.exec()
        else:
            pass

    def download_results(self):
        if (not self.test_data_loaded) or (not self.train_data_loaded):
            alert = QMessageBox()
            alert.setText('Need training and/or testing data. Please click ' +
                          '"Load training/testing data (.csv)" buttons to select data file.')
            alert.exec()
        elif self.model_results is None:
            alert = QMessageBox()
            alert.setText('No model results. Please click "Run model" button (after loading data).')
            alert.exec()
        else:
            model_int = 0 if self.model_choice == 'model 0' else 1
            results_file_name = f'results_model_{model_int}.csv'
            path = QFileDialog.getSaveFileName(self, 'Save results (csv)', results_file_name, 'CSV Files (*.csv)')
            if path != ('', ''):
                self.model_results['predictions'].to_csv(path[0], index=False)
                if os.path.isfile(path[0]):
                    alert = QMessageBox()
                    alert.setText('results saved to' + path[0])
                    alert.exec()


if __name__ == '__main__':
    appctxt = ApplicationContext()
    window = MainWindow()
    window.show()
    exit_code = appctxt.app.exec_()
    sys.exit(exit_code)
