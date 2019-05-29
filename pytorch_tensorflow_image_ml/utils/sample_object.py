
class SampleObject(object):
    def __init__(self, x, y, pred_y, score: float, layer_activations: dict=None, reversed_order=False):
        """
        Object interface for handling samples.
        The key feature of this is being able to sort these objects in a Queue via the `__lt__` method.

        Big note, if `reversed_order` is False, then calling the a get method in a priority queue will return the
        LOWEST score. The goal is that a Priority Queue of Sample objects will end up keeping the largest
        scores via removing the lowest ones during get calls.

        If `reversed_order` is True then the opposite will happen.

        Args:
            x:
            score: This is the score of that sample. Expected to be something like a probability output.
            You can then sort samples by their worst scores, and best scores.
        """
        self.x = x
        self.y = y
        self.layer_activations = layer_activations
        self.pred_y = pred_y
        self.reversed_order = reversed_order
        self.score = score

    def __lt__(self, other):
        return self.score < other.score if self.reversed_order else self.score > other.score

