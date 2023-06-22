# tftm.py
#
# Thin Film Thickness Measurement (TFTM) main file
# David Lister
# Started January 2023
#

from PySide6 import QtGui, QtCore
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QGridLayout, QFileDialog, QSlider, QLabel
import pyqtgraph as pg
import sys
# Setup Logging
import logging
import common
import analysis


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
        self.thickness_slider_fit.setTracking(False)
        self.thickness_slider_fit.valueChanged.connect(self.get_fit_slider_pos)

        self.thickness_slider_theory = QSlider(QtCore.Qt.Horizontal)
        self.thickness_slider_theory.setMinimum(1)
        self.thickness_slider_theory.setMaximum(3000)
        self.thickness_slider_theory.setValue(1000)
        self.thickness_slider_theory.valueChanged.connect(self.get_theory_slider_pos)


        self.amplitude_slider_theory = QSlider(QtCore.Qt.Horizontal)
        self.amplitude_slider_theory.setMinimum(0.1)
        self.amplitude_slider_theory.setMaximum(100)
        self.amplitude_slider_theory.setValue(1)
        self.amplitude_slider_theory.valueChanged.connect(self.get_theory_amplitude_slider_pos)

        self.thickness_display = QLabel()
        self.thickness_display.setText(f"Fit Thickness: {self.data.thickness} nm")

        self.thickness_theory_display = QLabel()
        self.thickness_theory_display.setText(f"Theoretical Model Thickness: {self.data.thickness_theoretical} nm")

        self.amplitude_theory_display = QLabel()
        self.amplitude_theory_display.setText(f"Theoretical Model Amplitude: {self.data.amplitude_theoretical}")

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
        self.layout.addWidget(self.thickness_slider_fit, 15, 0, 1, 7)
        self.layout.addWidget(self.thickness_display, 16, 3)
        self.layout.addWidget(self.thickness_slider_theory, 17, 0, 1, 7)
        self.layout.addWidget(self.thickness_theory_display, 18, 3)
        self.layout.addWidget(self.amplitude_slider_theory, 19, 0, 1, 7)
        self.layout.addWidget(self.amplitude_theory_display, 20, 3)

        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.setCentralWidget(self.widget)

    def get_calibration_file(self):
        self.logger.debug("Getting calibration file")
        self.state.calibration_file_path = QFileDialog.getOpenFileName()[0]
        self.logger.info(f"Calibration file set to '{self.state.calibration_file_path}'")
        self.data.calibration_spectrum = analysis.load_spectrum(self.state.calibration_file_path,
                                                                self.data.dark_reference_spectrum,
                                                                common.SUBTRACT_DARK_FROM_CALIBRATION)
        if self.data.calibration_spectrum is not None:
            self.logger.debug("Calibration loaded")
            self.calibration_plot.clear()
            self.calibration_plot.plot(self.data.calibration_spectrum[0], self.data.calibration_spectrum[1], pen=self.main_pen)
            self.draw_calculated_reflectance()

        else:
            self.logger.info("Error Loading calibration file")

    def get_reflectance_file(self):
        self.logger.debug("Getting reflectance spectrum file")
        self.state.reflectance_file_path = QFileDialog.getOpenFileName()[0]
        self.reflectance_display.setText(f"Reflectance file:{self.state.reflectance_file_path}")
        self.logger.info(f"Reflectance file set to '{self.state.reflectance_file_path}'")
        self.data.raw_reflectance_spectrum = analysis.load_spectrum(self.state.reflectance_file_path,
                                                                    self.data.dark_reference_spectrum,
                                                                    common.SUBTRACT_DARK_FROM_REFLECTED)
        if self.data.raw_reflectance_spectrum is not None:
            self.logger.debug("Raw reflectance loaded")
            self.raw_reflectance_plot.clear()
            self.raw_reflectance_plot.plot(self.data.raw_reflectance_spectrum[0], self.data.raw_reflectance_spectrum[1], pen=self.main_pen)
            self.draw_calculated_reflectance()

        else:
            logger.info("Error loading reflectance file")

    def get_dark_reference_file(self):
        self.logger.debug("Getting dark reference spectrum file")
        self.state.dark_reference_file_path = QFileDialog.getOpenFileName()[0]
        self.logger.info(f"Dark reference file set to '{self.state.dark_reference_file_path}'")
        self.data.dark_reference_spectrum = analysis.load_spectrum(self.state.dark_reference_file_path)
        self.logger.debug("Dark reference loaded")
        self.button_calibration_selection.setEnabled(True)
        self.button_reflectance_selection.setEnabled(True)
        self.draw_calculated_reflectance()

    def draw_calculated_reflectance(self):
        self.logger.debug("Drawing calculated reflectance")
        if analysis.can_calculate_reflectance(self.data):
            self.data.calc_reflectance_spectrum = analysis.calculate_reflectance(self.data)
            if self.state.last_thickness_fit != self.data.thickness:
                self.calc_fit()
                self.state.last_thickness_fit = self.data.thickness
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
        self.state.thin_film_properties_path = QFileDialog.getOpenFileName()[0]
        if self.state.thin_film_properties_path is not None:
            self.data.thin_film_optical_properties = analysis.load_optical_properties(self.state.thin_film_properties_path)
            logger.debug(f"Loaded {self.state.thin_film_properties_path} as thin film")
            self.draw_calculated_reflectance()

    def get_substrate_material(self):
        logger.debug("Loading substrate material properties")
        self.state.substrate_properties_path = QFileDialog.getOpenFileName()[0]
        if self.state.substrate_properties_path is not None:
            self.data.substrate_optical_properties = analysis.load_optical_properties(self.state.substrate_properties_path)
            logger.debug(f"Loaded {self.state.substrate_properties_path} as substrate")
            self.draw_calculated_reflectance()

    def calc_fit(self):
        logger.debug("Calculating fit")
        if analysis.can_calculate_thickness(self.data):
            logger.debug("Calculating thickness")
            self.data.fit = analysis.calculate_thickness(self.data)
            self.data.thickness = self.data.fit.params[0]
            self.set_fit_slider_pos()

    def get_fit_slider_pos(self, val):
        self.data.thickness = val
        self.draw_calculated_reflectance()
        self.thickness_display.setText(f"Thickness: {self.data.thickness:0.0f} nm")

    def set_fit_slider_pos(self):
        self.thickness_slider_fit.setValue(self.data.thickness)
        self.thickness_display.setText(f"Thickness: {self.data.thickness:0.0f} nm")

    def get_theory_slider_pos(self, val):
        self.data.thickness_theoretical = val
        self.thickness_theory_display.setText(f"Theoretical Model Thickness: {self.data.thickness_theoretical:0.0f} nm")
        self.draw_calculated_reflectance()

    def get_theory_amplitude_slider_pos(self, val):
        self.data.amplitude_theoretical = val
        self.amplitude_theory_display.setText(f"Theoretical Model Amplitude: {self.data.amplitude_theoretical:0.0f}")
        self.draw_calculated_reflectance()

    def draw_theoretical_fit(self):
        theo = lambda x: analysis.reflectance_model(x, 1,
                                                    self.data.thin_film_optical_properties.n,
                                                    self.data.thin_film_optical_properties.k,
                                                    self.data.substrate_optical_properties.n,
                                                    self.data.substrate_optical_properties.k,
                                                    self.data.thickness_theoretical, self.data.amplitude_theoretical)
        self.calc_reflectance_plot.plot(self.data.fit.wavelength, theo(self.data.fit.wavelength), pen=self.fit_pen)

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
