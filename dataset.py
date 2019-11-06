import numpy as np
import random
import matplotlib.pyplot as plt


class ECGDataset:
    def __init__(self, file_path):
        """
        :rtype: ECGDataset
        :param file_path: A file containing a numpy array with electrode data.
        """

        # Load data and create electrode index mapping
        self.file_path = file_path
        self.data = np.load(self.file_path) * 0.000625
        self.mapping = np.arange(0, len(self.data), 1)

        # Remove redundant electrodes (corners)
        self.data = np.delete(self.data, [0, 7, 127, 120], axis=0)
        self.mapping = np.delete(self.mapping, [0, 7, 127, 120], axis=0)

        # Declare peaks variable.
        self.peaks = None


def invert(dataset):
    # Multiply by -1 to invert data
    dataset.data = dataset.data * -1
    return dataset


def zoom(dataset, time_range):
    # Loop through all electrodes
    # and slice on x-axis to create zoom effect

    res = []
    for count in range((len(dataset.data))):
        res.append(dataset.data[count][time_range[0]: time_range[1]])

    dataset.data = np.array(res)
    return dataset


def sample(dataset, size=0.1):
    # Generate random indices based on input size
    # and use those indices to get a sample of the dataset.

    s_indices = random.sample(range(0, len(dataset.data)), int(size * len(dataset.data)))
    dataset.mapping = dataset.mapping[s_indices]
    dataset.data = dataset.data[s_indices]
    if dataset.peaks is not None:
        dataset.peaks = dataset.peaks[s_indices]
    return dataset


def slice_d(dataset, start, end):
    # Slice the mapping and the electrode data from start to end

    dataset.mapping = dataset.mapping[start: end]
    dataset.data = dataset.data[start: end]
    if dataset.peaks is not None:
        dataset.peaks = dataset.peaks[start: end]
    return dataset


def index(dataset, idx=None):
    # Select a single electrode from the dataset

    if not idx:
        # If index is not specified, choose a random electrode
        idx = dataset.mapping[random.randint(0, len(dataset.data))]
    else:
        idx = np.where(dataset.mapping == idx)[0][0]

    dataset.mapping = np.array([dataset.mapping[idx]])
    dataset.data = np.array([dataset.data[idx]])

    if dataset.peaks is not None:
        dataset.peaks = np.array([dataset.peaks[idx]])
    return dataset


def steepest_gradient(dataset, amount=5):
    # Calculate the acceleration and select an amount of steepest points

    dataset.peaks = []
    for i in range(len(dataset.data)):
        # Calculate acceleration
        g = np.gradient(np.gradient(dataset.data[i]))

        res = []
        for j in range(amount):
            # Select steepest point and nullify surrounding points.
            res.append(g.argmin())
            g[res[j] - 50: res[j] + 400] = 0

        dataset.peaks.append(res)
    dataset.peaks = np.array(dataset.peaks)
    return dataset


def template_matching(dataset):
    # Determine P-waves with template matching.

    dataset = steepest_gradient(dataset, 1)
    indexer = np.arange(100)[None, :] + np.arange(4900)[:, None]

    res = []
    for i in range(len(dataset.data)):
        # Calculate normalized cross correlation with mean subtraction
        # Fancy indexing is used to implement window sliding

        template = dataset.data[i][dataset.peaks[i][0]-50:dataset.peaks[i][0]+50]
        template = template - template.mean()

        data = dataset.data[i][indexer]
        data = data - np.mean(data, axis=1)[:, None]

        numerator = np.sum(data * template, axis=1)

        denominator = np.sqrt(
            np.sum(np.square(data), axis=1) * np.sum(np.square(template))
        )
        output = np.divide(numerator, denominator)

        # Find the P-waves by intersecting a horizontal line
        # Horizontal line decreases with 1% till it finds at least 5 P-waves.
        for j in np.arange(1.0, 0.0, -0.01):
            peaks = np.where(output >= j)[0]
            peak_distances = np.where(np.diff(peaks) >= 50)[0]

            if len(peak_distances) >= 4:
                peaks = np.where(output >= j-0.02)[0]
                peak_distances = np.where(np.diff(peaks) >= 50)[0]

                distinct_peaks = np.append(peak_distances, [peak_distances[len(peak_distances) - 1] + 1])
                res.append(peaks[distinct_peaks] + 50)
                break

    dataset.peaks = np.array(res)
    return dataset


def plot(dataset):
    # Plot the electrodes and
    # mark the P-waves if peaks variable is initialized

    plt.style.use('ggplot')
    plt.figure(figsize=(8, 4), dpi=150)

    for idx in range(len(dataset.data)):
        plt.title("Electrode idx: " + str(dataset.mapping[idx]))

        plt.plot(dataset.data[idx], antialiased=True)

        if dataset.peaks is not None:
            for i in dataset.peaks[idx]:
                plt.axvline(i, c="g", linewidth=10, alpha=0.5)
        plt.show()
        plt.close()
    return dataset
