import numpy as np
import pandas as pd

def extract_features(signal):
    rms = np.sqrt(np.mean(signal**2))
    peak = np.max(np.abs(signal))
    peak_to_peak = np.ptp(signal)
    standard_deviation = np.std(signal)
    crest_factor = peak / rms

    return rms, peak, peak_to_peak, standard_deviation, crest_factor


sampling_rate = 1000
duration = 2

time = np.linspace(
    0,
    duration,
    sampling_rate * duration,
    endpoint=False
)

data = []

num_samples = 200

for i in range(num_samples):
    normal_signal = np.sin(2 * np.pi * 30 * time)
    noise = 0.1 * np.random.randn(len(time))
    signal = normal_signal + noise

    features = extract_features(signal)
    data.append([*features, 0])  # 0 = normal


for i in range(num_samples):
    normal_signal = np.sin(2 * np.pi * 30 * time)
    fault_signal = 0.4 * np.sin(2 * np.pi * 90 * time)
    noise = 0.1 * np.random.randn(len(time))
    signal = normal_signal + fault_signal + noise

    features = extract_features(signal)
    data.append([*features, 1])  # 1 = faulty


columns = [
    "rms",
    "peak",
    "peak_to_peak",
    "standard_deviation",
    "crest_factor",
    "label"
]

df = pd.DataFrame(data, columns=columns)

df.to_csv("vibration_features_dataset.csv", index=False)

print("Dataset created successfully.")
print(df.head())
print()
print(df["label"].value_counts())