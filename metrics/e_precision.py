import numpy as np
import pdb
class Metric():
    def __init__(self, k, **kwargs):
        self.k        = k
        self.requires = ['nearest_features', 'target_labels']
        self.name     = 'e_precision@{}'.format(k)

    def __call__(self, target_labels, k_closest_classes, **kwargs):
        precision_at_k = np.sum([np.sum([i == target for i in  recalled_predictions[:self.k]])/self.k for target, recalled_predictions in zip(target_labels, k_closest_classes) ])/len(target_labels)
        return precision_at_k