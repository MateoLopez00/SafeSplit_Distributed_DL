"""
Matplotlib utilities with graceful fallback for environments without display.

Provides a plt alias that works whether or not matplotlib is available,
eliminating the need for if/else checks on every plotting function.
"""

class DummyPlt:
    """Dummy matplotlib.pyplot that warns and no-ops when matplotlib isn't available."""
    
    def __getattr__(self, name):
        """Return a no-op function for any matplotlib function, with warning."""
        def no_op(*args, **kwargs):
            print("Install matplotlib in the notebook environment to render the image grid.")
        return no_op
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass


try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False    
    plt = DummyPlt()

__all__ = ["plt", "HAS_MPL"]
