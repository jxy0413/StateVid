"""StateVid: lightweight state-guided video modeling."""

__version__ = "0.1.0"
__all__ = ["StateVid", "build_model", "__version__"]


def build_model(*args, **kwargs):
    from .models import build_model as factory
    return factory(*args, **kwargs)


def __getattr__(name):
    if name == "StateVid":
        from .models import ChangeModel
        return ChangeModel
    raise AttributeError(name)
