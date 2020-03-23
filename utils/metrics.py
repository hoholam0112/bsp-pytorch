# Useful Metric classes for PyTorch mimicking tf.keras.metric API
import torch

class Accuracy():
    """ Update Accuracy in online """
    def __init__(self):
        self._num_samples = 0
        self._num_corrects = 0

    def reset_states(self):
        """ Reset internal state of a Metric class"""
        self.__init__()

    def update_states(self, y_pred, y_true):
        """ Update internal state of Metric class

        Args:
            y_pred (torch.Tensor): class probability or logits.
                2-d tensor of size [num_samples, num_classes].
            y_true (torch.Tensor): groudtruth class labels encoded with
                integer in [0, num_classes-1]. 1d-tensor of size [num_samples].

        Returns:
            None
        """
        self._num_samples += y_pred.size(0)
        y_pred_int = torch.argmax(y_pred, 1)
        self._num_corrects += torch.sum(y_pred_int == y_true.data)

    def result(self):
        """ Compute metric and return it """
        return float(self._num_corrects) / self._num_samples

class Mean():
    def __init__(self):
        self._num_samples = 0
        self._sum = 0.0

    def reset_states(self):
        self.__init__()

    def update_states(self, inputs):
        self._num_samples += inputs.size(0)
        self._sum = torch.sum(inputs)

    def result(self):
        return self._sum/self._num_samples

