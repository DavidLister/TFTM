# tftm.py
#
# Thin Film Thickness Measurement (TFTM) main file
# David Lister
# Started January 2023
#

from PySide6 import QtGui, QtCore
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QGridLayout, QFileDialog, QSlider,\
    QLabel, QSizePolicy
import pyqtgraph as pg
import sys
# Setup Logging
import logging
import common
import analysis
import numpy as np


logger = logging.getLogger("TFTM")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

logger.debug("Logger Started")


class InternalState:
    """
    Class to hold internal program state, separate from the gui.

    All variable initiated as None type.
    """
    def __init__(self):
        self.calibration_file_path = None
        self.reflectance_file_path = None
        self.dark_reference_file_path = None
        self.thin_film_properties_path = None
        self.substrate_properties_path = None
        self.last_thickness_fit = None


class MainWindow(QMainWindow):
    """
    Main GUI Class.
    """
    def __init__(self):
        super().__init__()
        self.state = InternalState()
        self.data = analysis.DataContainer()
        self.logger = logging.getLogger("TFTM.MainWindow")
        self.logger.debug("Main window started")

        self.setWindowTitle("TFTM")
        self.colour = self.palette().color(QtGui.QPalette.Window)
        self.main_pen = pg.mkPen(color=(20, 20, 20))
        self.fit_pen = pg.mkPen(color=(153, 0, 0))
        self.reference_pen = pg.mkPen(color=(12, 105, 201))
        self.header_font = QtGui.QFont("Sans Serif", 12)
        self.body_font = QtGui.QFont("Sans Serif", 12)

        self.layout = QGridLayout()

        self.button_calibration_selection = QPushButton("Select Calibration")
        self.button_calibration_selection.clicked.connect(self.get_calibration_file)
        self.button_calibration_selection.setEnabled(not common.SUBTRACT_DARK_FROM_CALIBRATION)

        self.button_reflectance_selection = QPushButton("Select Reflected Spectrum")
        self.button_reflectance_selection.clicked.connect(self.get_reflectance_file)
        self.button_reflectance_selection.setEnabled(not common.SUBTRACT_DARK_FROM_REFLECTED)

        self.reflectance_display = QLabel()
        self.reflectance_display.setText(f"Reflectance file:{self.state.reflectance_file_path}")

        self.button_dark_reference_selection = QPushButton("Select Dark Reference Spectrum")
        self.button_dark_reference_selection.clicked.connect(self.get_dark_reference_file)

        self.button_thin_film_material = QPushButton("Select Thin Film Material")
        self.button_thin_film_material.clicked.connect(self.get_thin_film_material)

        self.button_substrate_material = QPushButton("Select Substrate Material")
        self.button_substrate_material.clicked.connect(self.get_substrate_material)

        self.button_reflectance_save = QPushButton("Save Reflectance")
        self.button_reflectance_save.clicked.connect(self.save_reflectance)

        self.calibration_plot = pg.PlotWidget()
        self.calibration_plot.setBackground(self.colour)
        self.calibration_plot.setTitle("Calibration Spectrum")
        self.calibration_plot.setLabel("bottom", "Wavelength (nm)")

        self.raw_reflectance_plot = pg.PlotWidget()
        self.raw_reflectance_plot.setBackground(self.colour)
        self.raw_reflectance_plot.setTitle("Raw Reflected Spectrum")
        self.raw_reflectance_plot.setLabel("bottom", "Wavelength (nm)")

        self.calc_reflectance_plot = pg.PlotWidget()
        self.calc_reflectance_plot.setBackground(self.colour)
        self.calc_reflectance_plot.setTitle("Reflectance Spectrum")
        self.calc_reflectance_plot.setLabel("bottom", "Wavelength (nm)")

        self.thickness_slider_fit = QSlider(QtCore.Qt.Horizontal)
        self.thickness_slider_fit.setMinimum(1)
        self.thickness_slider_fit.setMaximum(3000)
        self.thickness_slider_fit.setValue(1000)
        self.thickness_slider_fit.valueChanged.connect(self.get_fit_slider_pos)

        self.amplitude_slider_fit = QSlider(QtCore.Qt.Horizontal)
        self.amplitude_slider_fit.setMinimum(1)
        self.amplitude_slider_fit.setMaximum(7000)
        self.amplitude_slider_fit.setValue(1000)
        self.amplitude_slider_fit.valueChanged.connect(self.get_fit_amplitude_slider_pos)

        self.offset_slider_fit = QSlider(QtCore.Qt.Horizontal)
        self.offset_slider_fit.setMinimum(-2000)
        self.offset_slider_fit.setMaximum(2000)
        self.offset_slider_fit.setValue(0)
        self.offset_slider_fit.valueChanged.connect(self.get_fit_offset_slider_pos)

        self.button_do_fit = QPushButton("Run Fit")
        self.button_do_fit.clicked.connect(self.calc_fit)

        self.n_modification_slider = QSlider(QtCore.Qt.Horizontal)
        self.n_modification_slider.setMinimum(750)
        self.n_modification_slider.setMaximum(1250)
        self.n_modification_slider.setValue(1000)
        self.n_modification_slider.valueChanged.connect(self.get_n_modification_slider_pos)

        self.k_modification_slider = QSlider(QtCore.Qt.Horizontal)
        self.k_modification_slider.setMinimum(0.1)
        self.k_modification_slider.setMaximum(2000)
        self.k_modification_slider.setValue(1000)
        self.k_modification_slider.setTickInterval(0.1)
        self.k_modification_slider.valueChanged.connect(self.get_k_modification_slider_slider_pos)

        self.banner_fit_params = QLabel()
        self.banner_fit_params.setText("Fit Parameters")
        self.banner_fit_params.setFont(self.header_font)
        self.banner_fit_params.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.thickness_slider_display = QLabel()
        self.thickness_slider_display.setText(f"Thickness: {self.data.slider_thickness:0.1f} nm")

        self.amplitude_slider_display = QLabel()
        self.amplitude_slider_display.setText(f"Amplitude: {self.data.slider_amplitude:0.0f}")

        self.offset_slider_display = QLabel()
        self.offset_slider_display.setText(f"Offset: {self.data.slider_offset:0.0f}")

        self.spacing_line = QWidget()
        self.spacing_line.setFixedHeight(2)
        self.spacing_line.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.spacing_line.setStyleSheet("background-color: #c0c0c0;")

        self.spacing_line_2 = QWidget()
        self.spacing_line_2.setFixedHeight(2)
        self.spacing_line_2.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.spacing_line_2.setStyleSheet("background-color: #c0c0c0;")

        self.spacing_line_3 = QWidget()
        self.spacing_line_3.setFixedHeight(2)
        self.spacing_line_3.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.spacing_line_3.setStyleSheet("background-color: #c0c0c0;")

        self.spacing_line_4 = QWidget()
        self.spacing_line_4.setFixedHeight(2)
        self.spacing_line_4.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.spacing_line_4.setStyleSheet("background-color: #c0c0c0;")

        self.banner_fit_modification = QLabel()
        self.banner_fit_modification.setText("Thin film optical property modification")
        self.banner_fit_modification.setFont(self.header_font)
        self.banner_fit_modification.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.n_modification_display = QLabel()
        self.n_modification_display.setText(f"Refractive index adjustment")

        self.k_modification_display = QLabel()
        self.k_modification_display.setText(f"Extinction coefficient adjustment")

        self.layout.addWidget(self.button_calibration_selection, 0, 0)
        self.layout.addWidget(self.button_reflectance_selection, 0, 1)
        self.layout.addWidget(self.button_dark_reference_selection, 0, 2)
        self.layout.addWidget(self.button_thin_film_material, 0, 4)
        self.layout.addWidget(self.button_substrate_material, 0, 5)
        self.layout.addWidget(self.button_reflectance_save, 0, 6)
        self.layout.addWidget(self.reflectance_display, 1, 1, 1, 4)
        self.layout.addWidget(self.calibration_plot, 2, 0, 3, 3)
        self.layout.addWidget(self.raw_reflectance_plot, 2, 4, 3, 3)
        self.layout.addWidget(self.calc_reflectance_plot, 7, 0, 5, 7)

        self.layout.addWidget(self.spacing_line, 15, 0, 1, 7)
        self.layout.addWidget(self.spacing_line_2, 16, 0, 1, 7)
        self.layout.addWidget(self.banner_fit_params, 17, 3)
        self.layout.addWidget(self.thickness_slider_fit, 18, 0, 1, 7)
        self.layout.addWidget(self.thickness_slider_display, 19, 3)
        self.layout.addWidget(self.amplitude_slider_fit, 20, 0, 1, 3)
        self.layout.addWidget(self.button_do_fit, 20, 3)
        self.layout.addWidget(self.offset_slider_fit, 20, 4, 1, 3)
        self.layout.addWidget(self.amplitude_slider_display, 21, 2)
        self.layout.addWidget(self.offset_slider_display, 21, 4)

        self.layout.addWidget(self.spacing_line_3, 22, 0, 1, 7)
        self.layout.addWidget(self.spacing_line_4, 23, 0, 1, 7)
        self.layout.addWidget(self.banner_fit_modification, 24, 3)
        self.layout.addWidget(self.n_modification_slider, 25, 0, 1, 3)
        self.layout.addWidget(self.k_modification_slider, 25, 4, 1, 3)
        self.layout.addWidget(self.n_modification_display, 26, 1)
        self.layout.addWidget(self.k_modification_display, 26, 5)

        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.setCentralWidget(self.widget)

    def get_calibration_file(self):
        self.logger.debug("Getting calibration file")
        test_path = QFileDialog.getOpenFileName()[0]
        test_file = analysis.load_spectrum(test_path,
                                           dark_spectrum= self.data.dark_reference_spectrum,
                                           dark_subtraction_enabled= common.SUBTRACT_DARK_FROM_CALIBRATION,
                                           normalize=False,
                                           calculate_flux=common.USE_FLUX)
        if test_file is not None:
            self.logger.debug("Calibration loaded")
            self.data.calibration_spectrum = test_file
            self.state.calibration_file_path = test_path
            self.logger.info(f"Calibration file set to '{self.state.calibration_file_path}'")
            self.calibration_plot.clear()
            self.calibration_plot.plot(self.data.calibration_spectrum[0], self.data.calibration_spectrum[1], pen=self.main_pen)
            self.draw_calculated_reflectance()

        else:
            self.logger.info("Error Loading calibration file")

    def get_reflectance_file(self):
        self.logger.debug("Getting reflectance spectrum file")
        test_path = QFileDialog.getOpenFileName()[0]
        self.reflectance_display.setText(f"Reflectance file:{test_path}")
        test_file = analysis.load_spectrum(test_path,
                                           dark_spectrum=self.data.dark_reference_spectrum,
                                           dark_subtraction_enabled=common.SUBTRACT_DARK_FROM_REFLECTED,
                                           normalize=False,
                                           calculate_flux=common.USE_FLUX)
        if test_file is not None:
            self.logger.debug("Raw reflectance loaded")
            self.data.raw_reflectance_spectrum = test_file
            self.state.reflectance_file_path = test_path
            self.logger.info(f"Reflectance file set to '{self.state.reflectance_file_path}'")
            self.raw_reflectance_plot.clear()
            self.raw_reflectance_plot.plot(self.data.raw_reflectance_spectrum[0], self.data.raw_reflectance_spectrum[1], pen=self.main_pen)
            self.draw_calculated_reflectance()

        else:
            logger.warning("Error loading reflectance file")

    def get_dark_reference_file(self):
        self.logger.debug("Getting dark reference spectrum file")
        test_path = QFileDialog.getOpenFileName()[0]
        test_file = analysis.load_spectrum(test_path)
        if test_file is not None:
            self.logger.debug("Dark reference loaded")
            self.data.dark_reference_spectrum = test_file
            self.state.dark_reference_file_path = test_path
            self.logger.info(f"Dark reference file set to '{self.state.dark_reference_file_path}'")
            self.button_calibration_selection.setEnabled(True)
            self.button_reflectance_selection.setEnabled(True)
            self.draw_calculated_reflectance()

        else:
            self.logger.warning("Error loading dark reference file")

    def draw_calculated_reflectance(self):
        self.logger.debug("Drawing calculated reflectance")
        if analysis.can_calculate_reflectance(self.data):
            self.data.calc_reflectance_spectrum = analysis.calculate_reflectance(self.data,
                                                                                 limit_small_values=common.SET_SMALL_VALUES_TO_1,
                                                                                 normalize_reflectance=common.NORMALIZE_REFLECTANCE,
                                                                                 limit_to_peaks_in_bounds=common.LIMIT_TAILS_TO_PEAKS_IN_BOUNDS)
            # if self.state.last_thickness_fit != self.data.slider_thickness:
            #     self.calc_fit()
            #     self.state.last_thickness_fit = self.data.slider_thickness
            self.calc_reflectance_plot.clear()
            self.calc_reflectance_plot.plot(self.data.calc_reflectance_spectrum[0], self.data.calc_reflectance_spectrum[1], pen=self.main_pen)
            if self.data.fit is not None:
                logger.info(f"Calculated thickness is {self.data.fit.params[0]} +- {self.data.fit.error[0]}")
                self.calc_reflectance_plot.plot(self.data.fit.wavelength,
                                                self.data.fit.reflectance, pen=self.fit_pen)
            self.draw_theoretical_fit()
        else:
            self.logger.debug("Can't calculate reflectance")

    def get_thin_film_material(self):
        logger.debug("Loading thin film material properties")
        test_path = QFileDialog.getOpenFileName()[0]
        if test_path is not None and test_path != "":
            self.state.thin_film_properties_path = test_path
            self.data.thin_film_optical_properties = analysis.load_optical_properties(self.state.thin_film_properties_path)
            logger.debug(f"Loaded {self.state.thin_film_properties_path} as thin film")
            self.draw_calculated_reflectance()

        else:
            logger.info("Error: Didn't load thin film properties")

    def get_substrate_material(self):
        logger.debug("Loading substrate material properties")
        test_path = QFileDialog.getOpenFileName()[0]
        if test_path is not None and test_path != "":
            self.state.substrate_properties_path = test_path
            self.data.substrate_optical_properties = analysis.load_optical_properties(self.state.substrate_properties_path)
            logger.debug(f"Loaded {self.state.substrate_properties_path} as substrate")
            self.draw_calculated_reflectance()
        else:
            logger.info("Error: Didn't load substrate properties")
    def calc_fit(self):
        logger.debug("Calculating fit")
        if analysis.can_calculate_thickness(self.data):
            logger.debug("Calculating thickness")
            self.data.fit = analysis.calculate_thickness(self.data)
            thickness = self.data.fit.params[0]
            amplitude = self.data.fit.params[1]
            offset = self.data.fit.params[2]
            self.set_all_slider_pos(thickness, amplitude, offset)
            self.draw_calculated_reflectance()

    def get_fit_slider_pos(self, val):
        self.data.slider_thickness = val
        self.thickness_slider_display.setText(f"Thickness: {self.data.slider_thickness:0.1f} nm")
        self.draw_calculated_reflectance()

    def get_fit_amplitude_slider_pos(self, val):
        self.data.slider_amplitude = val/common.SLIDER_SCALE_FACTOR
        self.amplitude_slider_display.setText(f"Amplitude {self.data.slider_amplitude:0.2f}")
        self.draw_calculated_reflectance()

    def get_fit_offset_slider_pos(self, val):
        self.data.slider_offset = val/common.SLIDER_SCALE_FACTOR
        self.offset_slider_display.setText(f"Offset: {self.data.slider_offset:0.2f}")
        self.draw_calculated_reflectance()

    def set_all_slider_pos(self, thickness, amplitude, offset):
        self.data.slider_thickness = thickness
        self.data.slider_amplitude = amplitude
        self.data.slider_offset = offset

        self.thickness_slider_fit.setValue(self.data.slider_thickness)
        self.thickness_slider_display.setText(f"Thickness: {self.data.slider_thickness:0.1f} nm")

        self.amplitude_slider_fit.setValue(self.data.slider_amplitude * common.SLIDER_SCALE_FACTOR)
        self.amplitude_slider_display.setText(f"Amplitude: {self.data.slider_amplitude:0.1f} nm")

        self.offset_slider_fit.setValue(self.data.slider_offset * common.SLIDER_SCALE_FACTOR)
        self.offset_slider_display.setText(f"Offset: {self.data.slider_offset:0.1f} nm")

    def get_n_modification_slider_pos(self, val):
        if analysis.can_calculate_reflectance(self.data):
            self.data.n_modification = val/common.SLIDER_SCALE_FACTOR
            lower, upper = analysis.calculate_bounds(self.data)
            mask = (upper > self.data.calibration_spectrum[0]) * (self.data.calibration_spectrum[0] > lower)
            wavelengths = self.data.calc_reflectance_spectrum[0][mask]
            n_avg = np.mean(self.data.thin_film_optical_properties.n(wavelengths))

            self.n_modification_display.setText(
                f"Average refractive index: {n_avg * self.data.n_modification:0.2f} (Modification factor of {self.data.n_modification:0.2f})")
            self.draw_calculated_reflectance()

        else:
            pass

    def get_k_modification_slider_slider_pos(self, val):
        if analysis.can_calculate_reflectance(self.data):
            self.data.k_modification = val/common.SLIDER_SCALE_FACTOR
            lower, upper = analysis.calculate_bounds(self.data)
            mask = (upper > self.data.calibration_spectrum[0]) * (self.data.calibration_spectrum[0] > lower)
            wavelengths = self.data.calc_reflectance_spectrum[0][mask]
            k_avg = np.mean(self.data.thin_film_optical_properties.k(wavelengths))

            self.k_modification_display.setText(f"Average extinction coefficient: {k_avg * self.data.k_modification:0.2f} (Modification factor of {self.data.k_modification:0.2f})")
            self.draw_calculated_reflectance()

        else:
            pass

    def draw_theoretical_fit(self):
        theo = lambda x: analysis.reflectance_model(x, common.N_AIR,
                                                    self.data.thin_film_optical_properties.n,
                                                    self.data.thin_film_optical_properties.k,
                                                    self.data.substrate_optical_properties.n,
                                                    self.data.substrate_optical_properties.k,
                                                    self.data.slider_thickness,
                                                    self.data.slider_amplitude,
                                                    self.data.slider_offset,
                                                    n_factor=self.data.n_modification,
                                                    k_factor=self.data.k_modification)
        lower, upper = analysis.calculate_bounds(self.data)
        mask = (self.data.calibration_spectrum[0] < upper) * (self.data.calibration_spectrum[0] > lower)
        self.calc_reflectance_plot.plot(self.data.calibration_spectrum[0][mask], theo(self.data.calibration_spectrum[0][mask]), pen=self.reference_pen)

    def save_reflectance(self):
        logger.debug("Saving reflectance")
        fname = QFileDialog.getSaveFileName(filter="*.csv")[0]
        if len(fname) > 1:
            if self.data.calc_reflectance_spectrum is not None:
                out = self.data.calc_reflectance_spectrum.transpose()
                np.savetxt(fname, out, delimiter=',')
                logger.debug(f"Reflectance saved to {fname}")

            else:
                logger.debug("No data available to save")
        else:
            logger.debug("Invalid filename")



if __name__ == "__main__":
    app = QApplication([])

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
