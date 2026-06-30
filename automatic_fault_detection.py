import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("images", exist_ok=True)

# ------------------------------------
# Generate Synthetic Data
# ------------------------------------

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

# ------------------------------------
# FFT
# ------------------------------------

fft = np.fft.fft(fault_signal)

frequency = np.fft.fftfreq(
    len(fault_signal),
    d=1 / sampling_rate
)

magnitude = np.abs(fft)

positive = frequency >= 0

frequency = frequency[positive]
magnitude = magnitude[positive]

# ------------------------------------
# Automatic Fault Detection
# ------------------------------------

fault_band = (frequency > 80) & (frequency < 100)

fault_peak = np.max(magnitude[fault_band])

threshold = 150

print()

print("----------- MACHINE STATUS -----------")

if fault_peak > threshold:

    print("⚠ Fault Detected")

else:

    print("✓ Machine Operating Normally")

print()

print(f"Fault Peak Amplitude = {fault_peak:.2f}")

# ------------------------------------
# Plot
# ------------------------------------

plt.figure(figsize=(10,6))

plt.plot(
    frequency,
    magnitude
)

plt.axvspan(
    80,
    100,
    alpha=0.2,
    label="Fault Band"
)

plt.title("Automatic Fault Detection")

plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")

plt.grid(True)

plt.legend()

plt.xlim(0,150)

plt.savefig("images/automatic_fault_detection.png", dpi=300)

plt.show()