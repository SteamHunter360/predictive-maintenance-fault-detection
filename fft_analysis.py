import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("images", exist_ok=True)



# ----------------------------
# Generate Synthetic Signals
# ----------------------------

sampling_rate = 1000
duration = 2

time = np.linspace(
    0,
    duration,
    sampling_rate * duration,
    endpoint=False
)

normal_signal = np.sin(2 * np.pi * 30 * time)

fault_signal = (
    normal_signal
    + 0.4 * np.sin(2 * np.pi * 90 * time)
)

noise = 0.1 * np.random.randn(len(time))

fault_signal += noise

# ----------------------------
# FFT
# ----------------------------

fft = np.fft.fft(fault_signal)

frequency = np.fft.fftfreq(
    len(fault_signal),
    d=1 / sampling_rate
)

magnitude = np.abs(fft)

# Keep only positive frequencies
positive = frequency >= 0

frequency = frequency[positive]
magnitude = magnitude[positive]

# ----------------------------
# Plot
# ----------------------------

plt.figure(figsize=(10,6))

plt.plot(
    frequency,
    magnitude
)

plt.title("Frequency Spectrum (FFT)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")

plt.grid(True)

plt.xlim(0,150)

plt.savefig("images/fft_frequency_spectrum.png", dpi=300)

plt.show()