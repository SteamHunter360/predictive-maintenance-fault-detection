import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("images", exist_ok=True)
# Time settings
sampling_rate = 1000
duration = 2
time = np.linspace(0, duration, sampling_rate * duration)

# Normal machine vibration
normal_frequency = 30
normal_amplitude = 1.0

normal_signal = normal_amplitude * np.sin(
    2 * np.pi * normal_frequency * time
)

# Faulty machine vibration
fault_frequency = 90
fault_amplitude = 0.4

fault_signal = normal_signal + fault_amplitude * np.sin(
    2 * np.pi * fault_frequency * time
)

# Add random noise
noise = 0.1 * np.random.randn(len(time))

normal_signal_noisy = normal_signal + noise
fault_signal_noisy = fault_signal + noise

# Plot signals
plt.figure(figsize=(10, 6))

plt.plot(time, normal_signal_noisy, label="Normal Machine")
plt.plot(time, fault_signal_noisy, label="Faulty Machine")

plt.title("Synthetic Vibration Signal: Normal vs Faulty Machine")
plt.xlabel("Time (s)")
plt.ylabel("Vibration Amplitude")
plt.legend()
plt.grid(True)

plt.savefig("images/time_domain_vibration_signal.png", dpi=300)

plt.show()