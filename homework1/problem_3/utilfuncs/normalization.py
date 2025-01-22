import numpy as np


class Normalization:
    def __init__(self, normalization_type: str, data: np.ndarray):
        self.normalization_type = normalization_type
        self.raw_data = data

    def shift_and_scale(self):
        """
        Apply shift-and-scale normalization:
        Subtract the minimum value and divide by the maximum of the shifted values.
        Maps all values into the range [shift, shift + scale].
        """
        # get min and max values in array
        min_val = np.min(self.raw_data)
        max_val = np.max(self.raw_data)

        # subtract min from data and then divide by max - min
        normalized_data = ((self.raw_data - min_val) / (max_val - min_val))
        return normalized_data

    def zero_mean_unit_variance(self):
        """
        Normalize data to have zero mean and unit variance:
        Subtract the mean and divide by the standard deviation for each feature.
        """
        mean = np.mean(self.raw_data)
        std = np.std(self.raw_data)
        normalized_data = (self.raw_data - mean) / std
        return normalized_data

    def term_frequency(self):
        """
        Apply Term-Frequency (TF) normalization:
        Normalize each row by dividing values by the sum of the row.
        Assumes the data is text-based (e.g., term-document matrix).
        """
        row_sums = np.sum(self.raw_data, axis=1, keepdims=True)
        normalized_data = self.raw_data / row_sums
        return normalized_data

    def normalize(self, **kwargs):
        """
        Normalize data based on the selected normalization type.
        Accepts additional arguments for specific normalization methods.
        """
        if self.normalization_type == "shift_and_scale":
            return self.shift_and_scale(**kwargs)
        elif self.normalization_type == "zero_mean_unit_variance":
            return self.zero_mean_unit_variance()
        elif self.normalization_type == "term_frequency":
            return self.term_frequency()
        else:
            raise ValueError(f"Unknown normalization type: {self.normalization_type}")


# Example usage:
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Zero Mean Unit Variance Normalization
normalizer = Normalization(normalization_type="zero_mean_unit_variance", data=data)
normalized_data = normalizer.normalize()
print("Zero Mean Unit Variance Normalized Data:\n", normalized_data)

# Term Frequency Normalization
normalizer = Normalization(normalization_type="term_frequency", data=data)
normalized_data = normalizer.normalize()
print("Term Frequency Normalized Data:\n", normalized_data)

# Shift and Scale Normalization
normalizer = Normalization(normalization_type="shift_and_scale", data=data)
normalized_data = normalizer.normalize()
print("Shift and Scale Normalized Data:\n", normalized_data)