"""train.stop_step: stop at a step without shortening the LR schedule; refuse a stop_step the resume already passed."""
import dataclasses
import importlib.util
import sys

import pytest

from conftest import ROOT


def load_train_script():
    name = "train_lingbotvla"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "tasks/vla/train_lingbotvla.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def test_stop_step_defaults_off():
    train = load_train_script()
    fields = {f.name: f for f in dataclasses.fields(train.MyTrainingArguments)}
    assert fields["stop_step"].default is None


def test_check_stop_step():
    check = load_train_script().check_stop_step
    check(None, 0)
    check(None, 5000)
    check(2147, 0)
    check(4294, 2147)
    for stop, step in ((2147, 2147), (2147, 3000)):
        with pytest.raises(ValueError, match="stop_step"):
            check(stop, step)
