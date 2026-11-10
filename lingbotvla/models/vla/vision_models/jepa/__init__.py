"""Lazy import: the teacher pulls in torch, which the index builder never needs."""


def __getattr__(name):
    if name == "VJepaTeacher":
        from .teacher import VJepaTeacher
        return VJepaTeacher
    raise AttributeError(name)
