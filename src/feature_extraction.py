import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Generate Synthetic Signal
# -----------------------------

sampling_rate = 1000
duration = 2

time = np.linspace(
    0,
    duration,
    sampling_rate * duration,
    endpoint=False
)

signal = (
    np.sin(2*np.pi*30*time)
    + 0.4*np.sin(2*np.pi*90*time)
    + 0.1*np.random.randn(len(time))
)

# -----------------------------
# Feature Extraction
# -----------------------------

rms = np.sqrt(np.mean(signal**2))

peak = np.max(np.abs(signal))

peak_to_peak = np.ptp(signal)

standard_deviation = np.std(signal)

crest_factor = peak / rms

print("\n----- Extracted Features -----\n")

print(f"RMS               : {rms:.4f}")
print(f"Peak              : {peak:.4f}")
print(f"Peak-to-Peak      : {peak_to_peak:.4f}")
print(f"Standard Deviation: {standard_deviation:.4f}")
print(f"Crest Factor      : {crest_factor:.4f}")

# -----------------------------
# Plot
# -----------------------------

plt.figure(figsize=(10,5))

plt.plot(time, signal)

plt.title("Synthetic Vibration Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.grid(True)

plt.show()