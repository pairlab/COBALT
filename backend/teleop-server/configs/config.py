"""
Defines a base config class from which all other configs should be defined.
"""

import six  # preserve metaclass compatibility between python 2 and 3
import json
import copy

# global dictionary for remembering name - class mappings
REGISTERED_CONFIGS = {}


def register_config(target_class):
    """
    Registers a config class in the global registry.

    :param target_class: the class to register
    """
    REGISTERED_CONFIGS[target_class.__name__] = target_class


def make_config(config_name, *args, **kwargs):
    """
    Creates an instance of a config. Make sure to pass any other needed arguments.

    :param config_name: name for the config class
    """
    if config_name not in REGISTERED_CONFIGS:
        raise Exception(f"Config {config_name} not found. Make sure it is a registered config among: {', '.join(REGISTERED_CONFIGS)}")
    return REGISTERED_CONFIGS[config_name](*args, **kwargs)


class CCRConfigMeta(type):
    """
    Define a metaclass for constructing a config class.
    It registers configs into the global registry.
    """

    def __new__(meta, name, bases, class_dict):
        cls = super(CCRConfigMeta, meta).__new__(meta, name, bases, class_dict)
        register_config(cls)
        return cls


@six.add_metaclass(CCRConfigMeta)
class CCRConfig(dict):
    """
    Base class for all configs.
    """

    def __init__(__self, *args, **kwargs):
        object.__setattr__(__self, '__finalized', False)
        object.__setattr__(__self, '__parent', kwargs.pop('__parent', None))
        object.__setattr__(__self, '__key', kwargs.pop('__key', None))
        for arg in args:
            if not arg:
                continue
            elif isinstance(arg, dict):
                for key, val in arg.items():
                    __self[key] = __self._hook(val)
            elif isinstance(arg, tuple) and (not isinstance(arg[0], tuple)):
                __self[arg[0]] = __self._hook(arg[1])
            else:
                for key, val in iter(arg):
                    __self[key] = __self._hook(val)

        for key, val in kwargs.items():
            __self[key] = __self._hook(val)

        # Add name as well if this is top level class
        if object.__getattribute__(__self, '__parent') is None:
            __self["config_class"] = __self._hook(type(__self).__name__)

    def _finalize(self):
        """
        Lock the addict to avoid silent failures due to addict retuning empty dict rather than
        throw a key not found error.
        """
        object.__setattr__(self, '__finalized', True)

        for k in self:
            if isinstance(self[k], CCRConfig):
                self[k]._finalize()

    def __setattr__(self, name, value):
        if object.__getattribute__(self, '__finalized'):
            raise RuntimeError(f'This addict has been finalized and {name} is not in this addict')

        if hasattr(CCRConfig, name):
            raise AttributeError(f"'Dict' object attribute '{name}' is read-only")
        else:
            self[name] = value

    def __setitem__(self, name, value):
        super(CCRConfig, self).__setitem__(name, value)
        p = object.__getattribute__(self, '__parent')
        key = object.__getattribute__(self, '__key')
        if p is not None:
            p[key] = self

    def __reduce__(self):
        """Custom reduce method for pickling"""
        return (self.__class__, (dict(self),))  # Return a reconstructable state

    def __add__(self, other):
        if not self.keys():
            return other
        else:
            self_type = type(self).__name__
            other_type = type(other).__name__
            raise TypeError(f"unsupported operand type(s) for +: '{self_type}' and '{other_type}'")

    @classmethod
    def _hook(cls, item):
        if isinstance(item, dict):
            return cls(item)
        elif isinstance(item, (list, tuple)):
            return type(item)(cls._hook(elem) for elem in item)
        return item

    def __getattr__(self, item):
        return self.__getitem__(item)

    def __repr__(self):
        json_string = json.dumps(self.to_dict(), indent=4)
        return json_string

    def __getitem__(self, name):
        if name not in self:
            if object.__getattribute__(self, '__finalized'):
                 raise RuntimeError(f'This addict has been finalized and {name} is not in this addict')
            return CCRConfig(__parent=self, __key=name)
        return super(CCRConfig, self).__getitem__(name)

    def __delattr__(self, name):
        del self[name]

    def to_dict(self):
        base = {}
        for key, value in self.items():
            if isinstance(value, type(self)):
                base[key] = value.to_dict()
            elif isinstance(value, (list, tuple)):
                base[key] = type(value)(item.to_dict() if isinstance(item, type(self)) else item for item in value)
            else:
                base[key] = value
        return base

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)

    def __deepcopy__(self, memo):
        other = self.__class__()
        memo[id(self)] = other
        for key, value in self.items():
            other[copy.deepcopy(key, memo)] = copy.deepcopy(value, memo)
        return other

    def update(self, *args, **kwargs):
        other = {}
        if args:
            if len(args) > 1:
                raise TypeError()
            other.update(args[0])
        other.update(kwargs)
        for k, v in other.items():
            if (
                (k not in self)
                or (not isinstance(self[k], dict))
                or (not isinstance(v, dict))
            ):
                self[k] = v
            else:
                self[k].update(v)

    def safe_update(self, *args, **kwargs):
        other = {}
        if args:
            if len(args) > 1:
                raise TypeError()
            other.update(args[0])
        other.update(kwargs)
        for k, v in other.items():
            if k not in self:
                raise KeyError(f"safe_update(): key {k} does not exist in the config")
            if (not isinstance(self[k], dict)) or (not isinstance(v, dict)):
                self[k] = v
            else:
                self[k].safe_update(v)

    def __getnewargs__(self):
        return tuple(self.items())

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)

    def setdefault(self, key, default=None):
        if key in self:
            return self[key]
        else:
            self[key] = default
            return default

    def infer_settings(self):
        raise NotImplementedError

    def dump(self, filename=None):
        """
        Dumps the config to a json.
        If filename is not None, dump to file.
        Returns a string.
        """
        json_string = json.dumps(self.to_dict(), indent=4)
        if filename is not None:
            f = open(filename, "w")
            f.write(json_string)
            f.close()
        return json_string
