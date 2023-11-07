from time import perf_counter
from contextlib import contextmanager
from functools import wraps


@contextmanager
def catchtime() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start


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
