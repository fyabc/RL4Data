# -*- coding: utf-8 -*-

from __future__ import print_function


class NameRegister(object):
    """A helper class to select different class by string names or aliases in options."""

    # [NOTE] Subclasses MUST override this variable.
    NameTable = {}

    @classmethod
    def register_class(cls, aliases, clazz=None):
        clazz = cls if clazz is None else clazz

        # Default add class name (lower)
        cls.NameTable[clazz.__name__.lower()] = clazz
        for name in aliases:
            cls.NameTable[name] = clazz

    @classmethod
    def get_by_name(cls, name):
        # [NOTE] This must be implemented in subclasses because `eval` cannot run at here.
        return cls.NameTable[name.lower()]
