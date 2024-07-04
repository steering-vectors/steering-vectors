"""Useful design patterns."""

from typing import Generic, TypeVar

T = TypeVar("T")


# Reference: https://stackoverflow.com/a/7346105
class Singleton(Generic[T]):
    """
    A non-thread-safe helper class to ease implementing singletons.
    This should be used as a decorator -- not a metaclass -- to the
    class that should be a singleton.

    The decorated class can define one `__init__` function that
    takes only the `self` argument. Also, the decorated class cannot be
    inherited from. Other than that, there are no restrictions that apply
    to the decorated class.

    To get the singleton instance, use the `instance` method. Trying
    to use `__call__` will result in a `TypeError` being raised.


    Example:
    ```
    @Singleton
    class Foo:
    def __init__(self):
        print 'Foo created'

    f = Foo() # Error, this isn't how you get the instance of a singleton

    f = Foo.instance() # Good. Being explicit is in line with the Python Zen
    g = Foo.instance() # Returns already created instance

    print f is g # True
    ```

    """

    def __init__(self, decorated: T):
        self._decorated = decorated

    def instance(self) -> T:
        """
        Returns the singleton instance. Upon its first call, it creates a
        new instance of the decorated class and calls its `__init__` method.
        On all subsequent calls, the already created instance is returned.

        """
        try:
            return self._instance  # type: ignore
        except AttributeError:
            self._instance = self._decorated()  # type: ignore
            return self._instance

    def __call__(self) -> None:
        raise TypeError("Singletons must be accessed through `instance()`.")

    def __instancecheck__(self, inst: T) -> bool:
        return isinstance(inst, self._decorated)  # type: ignore
