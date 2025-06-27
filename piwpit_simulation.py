# -----------------------------------------
# Import necessary scientific computing libraries
# -----------------------------------------
import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt
import random
from sklearn import metrics
import seaborn as sns

# -----------------------------------------
# Define patient-specific tissue layer thicknesses in meters
# (converted from mm or cm where appropriate)
# These represent skin, fat, muscle, and bone layers
# across 7 patient profiles used to simulate anatomical variability.
# -----------------------------------------
skin_width=[69.6e-4, 80e-4, 90e-4, 100e-4, 110e-4, 121.9e-4, 1.89e-1]
fat_width=[0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.57]
muscle_width=[1.79, 1.9, 2.0, 2.12, 2.2, 2.5, 2.7]
bone_width=[0.65, 0.72, 0.76, 0.85, 1.312, 1.4, 1.588]

# -----------------------------------------
# Computes the patient-specific final pressure value after ultrasonic propagation
# using exponential attenuation for each profile.
# Attenuation coefficients:
# skin: 0.4 dB/cm/MHz, fat: 0.6, muscle: 1.44, at 0.7 MHz center frequency.
# -----------------------------------------
def patient_profiles():
    p_final = []
    for i in range(7):
        attenuation = ((0.4*0.7*skin_width[i]) + (0.6*0.7*fat_width[i]) + (1.44*0.7*muscle_width[i])) / 8.686
        p_value = (0.169e6) * np.exp(-attenuation) * 0.8855  # propagation, attenuation, reflection factors
        p_final.append(p_value)
    return p_final

# -----------------------------------------
# On-Off Keying (OOK) Modulation:
# Takes a single-cycle ultrasonic waveform and bitstream,
# modulates by including waveform for '1' and zeros for '0' with timing aligned to sampling_rate.
# -----------------------------------------
def modulate_ook(signal, bitstream, symbol_duration, sampling_rate):
    modulated_signal = []
    t = np.arange(0, len(bitstream) * symbol_duration, 1/sampling_rate)
    for bit in bitstream:
        if bit == 1:
            modulated_signal.extend(signal)
        else:
            modulated_signal.extend(np.zeros_like(signal))
    return np.array(modulated_signal), t

# -----------------------------------------
# On-Off Keying (OOK) Demodulation:
# Recovers bits by checking max amplitude within each symbol duration
# and comparing against a threshold.
# -----------------------------------------
def demodulate_ook(received_signal, symbol_duration, sampling_rate, threshold):
    symbols = int(len(received_signal) / (symbol_duration * sampling_rate))
    demodulated_bits = []
    for i in range(symbols):
        start_index = int(i * symbol_duration * sampling_rate) + 1
        end_index = int((i + 1) * symbol_duration * sampling_rate) - 1
        symbol_amplitude = np.max(np.abs(received_signal[start_index:end_index]))
        if symbol_amplitude > threshold + 0.001:
            demodulated_bits.append(1)
        else:
            demodulated_bits.append(0)
    return np.array(demodulated_bits)

# -----------------------------------------
# Simulation Parameters:
# f: carrier frequency (750 kHz),
# symbol_duration: 1 microsecond per symbol,
# sampling_rate: 100 MHz sampling for high-resolution waveform generation.
# -----------------------------------------
A = patient_profiles()
f = 750e3
symbol_duration = 1e-6
sampling_rate = 1e8

# -----------------------------------------
# Prepare containers for collecting simulation results
# across patients, noise conditions, and random bitstreams.
# -----------------------------------------
total_mod_bits, total_demod_bits = [], []
error_rates = []
avg_error_rate = []
noise = []
snr = 5

# -----------------------------------------
# Compute noise thresholds across patient profiles and SNR steps
# Noise scales inversely with SNR while thresholds are used during demodulation.
# -----------------------------------------
for i in range(7):
    for k in range(4):
        noise.append(A[i] / 10**(snr / 20))
        snr += 5
    snr = 5

vector1 = np.array(noise)
temp = []
for i in range(7):
    for k in range(4):
        temp.append(A[i])
vector2 = np.array(temp)
threshold = list(vector1 + vector2)

# -----------------------------------------
# Main simulation loop:
# For each patient and noise scenario:
# - generate random bitstreams (1-200 bits),
# - generate single-cycle ultrasonic waveform at desired amplitude,
# - modulate using OOK,
# - add AWGN,
# - demodulate,
# - compute bit error rates (BER).
# -----------------------------------------
noise_counter = 0
for amp in A:
    for k in range(4):
        for j in range(20):
            random_stream = [random.randint(0, 1) for _ in range(random.randint(1, 200))]
            bitstream = np.array(random_stream)
            total_mod_bits.extend(bitstream)

            # Generate single-cycle ultrasonic waveform
            t = np.arange(0, symbol_duration, 1/sampling_rate)
            ultrasonic_wave = amp * np.sin(2 * np.pi * f * t)

            modulated_signal, modulated_time = modulate_ook(ultrasonic_wave, bitstream, symbol_duration, sampling_rate)

            # Generate AWGN with fixed SNR (20 dB for simplicity)
            snr_db = 20
            noise_power = 10 ** (-snr_db / 10)
            noise_vector = np.random.normal(noise_power, noise_power / sqrt(2), len(modulated_signal))
            
            # Add noise to the modulated signal
            modulated_signal_with_noise = modulated_signal + noise[noise_counter]

            # Demodulate using threshold determined earlier
            demodulated_bits = demodulate_ook(modulated_signal_with_noise, symbol_duration, sampling_rate, threshold[noise_counter])
            total_demod_bits.extend(demodulated_bits)

            # Compute BER
            error = np.sum(demodulated_bits != bitstream)
            bit_error_rate = error / len(demodulated_bits) if len(demodulated_bits) > 0 else 0
            error_rates.append(bit_error_rate)
        noise_counter += 1

    avg_error_rate.extend(error_rates)
    error_rates = []

# -----------------------------------------
# Reshape error rates for boxplot visualization
# Each patient profile will have its BER distribution displayed.
# -----------------------------------------
ber_plot = []
for i in range(0, len(avg_error_rate), 80):
    ber_plot.append(avg_error_rate[i:i+80])

# -----------------------------------------
# Visualization: Boxplot of BER vs Patient Profiles
# Allows quick comparison of modulation robustness across anatomical variability.
# -----------------------------------------
plt.figure(figsize=(6, 4))
plt.boxplot(ber_plot)
plt.title('Bit Error Rate vs. Patient Profiles')
plt.xlabel('Patient Profiles')
plt.ylabel('Bit Error Rate')
plt.show()