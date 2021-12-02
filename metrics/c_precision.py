import numpy as np
class Metric():
    def __init__(self, k, **kwargs):
        self.k        = k
        self.requires = ['nearest_features_cosine', 'target_labels']
        self.name     = 'c_precision@{}'.format(k)

    def __call__(self, target_labels, k_closest_classes_cosine, **kwargs):
        precision_at_k = np.sum([np.sum([i == target for i in  recalled_predictions[:self.k]])/self.k for target, recalled_predictions in zip(target_labels, k_closest_classes_cosine) ])/len(target_labels)
        return precision_at_k
