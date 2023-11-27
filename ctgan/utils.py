from time import perf_counter
from contextlib import contextmanager
from functools import wraps


@contextmanager
def catchtime():
    start = perf_counter()
    yield lambda: perf_counter() - start


@contextmanager
def evaluating(net):
    """Temporarily switch to evaluation mode.

    Obtained from https://discuss.pytorch.org/t/opinion-eval-should-be-a-context-manager/18998/3
    Following snippet is licensed under MIT license.
    """
    istrain = net.training
    try:
        net.eval()
        yield net
    finally:
        if istrain:
            net.train()


def monitor_loss(name):
    """Parametrized decorator, name with be used to store loss function."""

    def _monitor(method):
        """Decorator to log loss functions"""

        @wraps(method)
        def _loss(self, *method_args, **method_kwargs):
            l = method(self, *method_args, **method_kwargs)
            if self.track_loss:
                self.loss_info[name] = l.detach().cpu().numpy()
            return l

        return _loss

    return _monitor
