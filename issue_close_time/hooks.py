from typing import Callable


class Hook:
    """
    The base class for hooks. User-defined hooks must be a
    derived class of this class. Hooks can be used in various
    places to introduce user-defined behavior in addition to
    the standard operations performed by the package.
    """

    def __init__(self, name: str, function: Callable):
        """
        Initializes the hook.
        Args:
        =====
        name: str - The name of the hook.
        function: Callable - The function that is run. This function
                will be passed *args and **kwargs.
        """
        self.name = name
        self.function = function

    def call(self, *args, **kwargs):
        self.function(*args, **kwargs)
