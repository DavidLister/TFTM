
# analysis.py
#
# Analysis-related code goes here
# David Lister
# January 2023
#

import numpy as np
from scipy import interpolate, odr, signal
import logging

import common

logger = logging.getLogger("TFTM.analysis")


class DataContainer:
    """
    Class for holding data.

    Variable initialized to None.
    """
    def __init__(self):
        self.calibration_spectrum = None
        self.raw_reflectance_spectrum = None
        self.dark_reference_spectrum = None
        self.calc_reflectance_spectrum = None
        self.theo_reflectance_spectrum = None
        self.thin_film_optical_properties = None
        self.substrate_optical_properties = None
        self.fit = None
        self.slider_thickness = 1000
        self.slider_amplitude = 1
        self.slider_offset = 0
        self.n_modification = 1
        self.k_modification = 1


class OpticalProperties:
    """
    Hold optical property data
    """
    def __init__(self, wavelength, n, k):
        self.lower = np.min(wavelength)
        self.upper = np.max(wavelength)
        self.n = interpolate.interp1d(wavelength, n)
        self.k = interpolate.interp1d(wavelength, k)


class Fit:
    """
    Holds fit data
    """
    def __init__(self, x, y, B, B_err):
        self.wavelength = x
        self.reflectance = y
        self.params = B
        self.error = B_err


def load_spectrum(fname, dark_spectrum=None, dark_subtraction_enabled=True, normalize=False, calculate_flux=False):
    logger.debug("Loading a spectrum file")
    if fname == "":
        logger.info("No file selected")
        return None
    data = np.loadtxt(fname,
                      delimiter=common.SPECTRUM_FILE_DELIMITER,
                      skiprows=common.SPECTRUM_FILE_SKIPLINES)

    with open(fname, 'r') as f:
        header = [f.readline() for i in range(common.SPECTRUM_FILE_HEADER_LENGTH)]

    count = len(data[0][common.SPECTRUM_DATA_START_ROW:])
    data = data.transpose()
    data_out = np.zeros((2, sum(data[common.SPECTRUM_DATA_START_ROW] > 0)))
    data_out[0] = data[common.SPECTRUM_WAVELENGTH_ROW][data[common.SPECTRUM_DATA_START_ROW] > 0]

    # Sum all the non-zero data from columns 4 to N and add it to the output data vector
    data_out[1] = np.mean(data[common.SPECTRUM_DATA_START_ROW:, data[common.SPECTRUM_DATA_START_ROW] > 0], axis=0)

    # dark_subtraction
    if dark_spectrum is not None and dark_subtraction_enabled:
        data_out[1] = data_out[1] - dark_spectrum[1]
        data_out[1][data_out[1] < 1] = 1  # For numerical stability.

    if calculate_flux:
        exposure_time_us = header[common.SPECTRUM_FILE_EXPOSURE_INDEX].split(':')[1][1:].split('µ')[0]
        data_out[1] = data_out[1] / (int(exposure_time_us) * 1e-6)  # Calculating flux in counts/second

    # Normalize the spectrum
    if normalize:
        data_out[1] = data_out[1] / np.max(data_out[1])

    return data_out


def can_calculate_reflectance(data_container):
    logger.debug("Can reflectance be calculated?")
    if not isinstance(data_container.calibration_spectrum, np.ndarray):
        logger.debug("No - Calibration spectrum is missing")
        return False

    if not isinstance(data_container.raw_reflectance_spectrum, np.ndarray):
        logger.debug("No - Raw reflectance spectrum missing")
        return False

    cal_shape = data_container.calibration_spectrum.shape
    refl_shape = data_container.raw_reflectance_spectrum.shape
    if cal_shape == refl_shape:
        logger.debug("Yes")
        return True

    else:
        logger.warning(f"Can't calculate reflectance, however two numpy arrays are loaded of shapes {cal_shape}, {refl_shape}")


def calculate_reflectance(data_container, limit_small_values=False, normalize_reflectance=True):
    logger.debug("Calculating reflectance")
    # Assumes array shapes have already been checked

    data = np.zeros(data_container.calibration_spectrum.shape)
    data[0] = data_container.calibration_spectrum[0]

    reflected = data_container.raw_reflectance_spectrum[1]
    calibration = data_container.calibration_spectrum[1]

    # limit_small_values
    if limit_small_values:
        mask = reflected < common.SMALL_VALUE_THRESHOLD
        mask = np.logical_or(mask, calibration < common.SMALL_VALUE_THRESHOLD)
        reflected[mask] = 1
        calibration[mask] = 1

    reflectance = reflected / calibration

    # normalize_reflectance
    if normalize_reflectance:
        reflectance = reflectance / np.max(reflectance)

    data[1] = reflectance

    # Normalize so max is 1
    data[1] = data[1] / np.max(data[1])
    return data


def load_optical_properties(fname):
    logger.debug(f"Loading optical properties for {fname}")
    data = np.loadtxt(fname, skiprows=common.OPTICAL_PROPS_SKIPLINES, delimiter=common.OPTICAL_PROPS_DELIMITER)
    data = data.transpose()
    return OpticalProperties(data[0] * 1000, data[1], data[2])  # 1000 is unit conversion


efield_reflectance = lambda n1, n2: (n1 - n2) / (n1 + n2)
efield_reflectance_complex = lambda n1, k1, n2, k2: ((n1 - 1j * k1) - (n2 - 1j * k2)) / ((n1 - 1j * k1) + (n2 - 1j * k2))


def reflectance_model(wavelengths, n_air, n_tf, k_tf, n_sub, k_sub, d, A, B, n_factor=1, k_factor=1):
    """
    Based on equations 3.4, 3.10 and 3.6 in "A Practical Guide to Optical Metrology for Thin Films by Quinten
    """
    r01 = efield_reflectance_complex(n_air, 0, n_tf(wavelengths)*n_factor, k_tf(wavelengths) * k_factor)
    r12 = efield_reflectance_complex(n_tf(wavelengths)*n_factor, k_tf(wavelengths) * k_factor, n_sub(wavelengths), k_sub(wavelengths))
    R01 = np.real(r01 * np.conj(r01))
    R12 = np.real(r12 * np.conj(r12))
    phase_shift = np.arctan(np.imag(np.conj(r01) * r12) / np.real(np.conj(r01) * r12))

    refl_numerator = R01 + R12 * np.exp(-(8 * np.pi / wavelengths) * k_tf(wavelengths) * k_factor * d) + \
                     2 * np.sqrt(R01 * R12) * np.exp(-(4 * np.pi / wavelengths) * k_tf(wavelengths) * k_factor * d) * np.cos((4 * np.pi / wavelengths) * n_tf(wavelengths) * n_factor * d + phase_shift)

    refl_denom = 1 + R01 * R12 * np.exp(-(8 * np.pi / wavelengths) * k_tf(wavelengths) * k_factor * d) + \
                     2 * np.sqrt(R01 * R12) * np.exp(-(4 * np.pi / wavelengths) * k_tf(wavelengths) * k_factor * d) * np.cos(
        (4 * np.pi / wavelengths) * n_tf(wavelengths) * n_factor * d + phase_shift)

    return A * refl_numerator / refl_denom + B


def simple_model(wavelengths, n_air, n_tf, k_tf, n_sub, k_sub, d, A, B):
    r01 = efield_reflectance_complex(n_air, 0, n_tf(wavelengths), k_tf(wavelengths))
    r12 = efield_reflectance_complex(n_tf(wavelengths), k_tf(wavelengths), n_sub(wavelengths), k_sub(wavelengths))
    R01 = np.real(r01 * np.conj(r01))
    R12 = np.real(r12 * np.conj(r12))
    phase_shift = np.arctan(np.imag(np.conj(r01) * r12) / np.real(np.conj(r01) * r12))
    return A * np.cos((4 * np.pi / wavelengths) * n_tf(wavelengths) * d + phase_shift) + B


def calculate_bounds(data_container):
    lower = max((data_container.thin_film_optical_properties.lower,
                 data_container.substrate_optical_properties.lower,
                 data_container.calibration_spectrum[0][0],
                 common.FIT_LOWER_BOUND_MINIMUM))

    upper = min((data_container.thin_film_optical_properties.upper,
                 data_container.substrate_optical_properties.upper,
                 data_container.calibration_spectrum[0][-1],
                 common.FIT_UPPER_BOUND_MAXIMUM))

    return lower, upper


def calculate_thickness(data_container):
    logger.debug("Calculating film thickness")
    # assumes data has been checked

    lower, upper = calculate_bounds(data_container)

    logger.debug(f"Bounds are: {lower}, {upper}")

    mask = (data_container.calibration_spectrum[0] < upper) * (data_container.calibration_spectrum[0] > lower)

    logger.debug(f"Min: {np.min(data_container.calibration_spectrum[0][mask])} -- Max: {np.max(data_container.calibration_spectrum[0][mask])}")

    def fit_model(B, x):
        """ B[0] is thickness, B[1] is scaling factor"""
        return reflectance_model(x, common.N_AIR,
                                 data_container.thin_film_optical_properties.n,
                                 data_container.thin_film_optical_properties.k,
                                 data_container.substrate_optical_properties.n,
                                 data_container.substrate_optical_properties.k,
                                 B[0], B[1], B[2])

    model = odr.Model(fit_model)
    fit_data = odr.RealData(data_container.calc_reflectance_spectrum[0][mask],
                            data_container.calc_reflectance_spectrum[1][mask]) # todo - No error at the moment

    fit = odr.ODR(fit_data, model, beta0=[data_container.slider_thickness,
                                          data_container.slider_amplitude,
                                          data_container.slider_offset])
    output = fit.run()
    output.pprint()

    fit_y = fit_model(output.beta, data_container.calc_reflectance_spectrum[0][mask])

    return Fit(data_container.calc_reflectance_spectrum[0][mask], fit_y, output.beta, output.sd_beta)


def can_calculate_thickness(data):
    if not can_calculate_reflectance(data):
        return False

    if data.thin_film_optical_properties is not None and data.substrate_optical_properties is not None:
        return True

    return False
