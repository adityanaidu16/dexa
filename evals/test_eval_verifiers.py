"""CPU unit tests for the pure verifiers in modal_eval_compute.py."""

import importlib.util
from pathlib import Path

# modal is installed; the module builds an Image spec at import (no GPU/vllm needed).
_SPEC = importlib.util.spec_from_file_location(
    "modal_eval_compute", Path(__file__).resolve().parent / "modal_eval_compute.py")
ev = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ev)


def test_extract_final_number():
    assert ev.extract_final_number("... so #### 42") == 42.0
    assert ev.extract_final_number("The answer is $1,200.") == 1200.0
    assert ev.extract_final_number("temperature dropped to -3.5 degrees") == -3.5
    assert ev.extract_final_number("no numbers here") is None
    assert ev.extract_final_number("he had 5 apples then 42 left") == 42.0


def test_majority():
    assert ev.majority([1, 2, 2, 3, 2]) == 2
    assert ev.majority([None, 7, 7, None]) == 7
    assert ev.majority([None, None]) is None


def test_extract_python():
    txt = "Here you go:\n```python\ndef f(x):\n    return x+1\n```\nDone."
    assert ev.extract_python(txt, "f") == "def f(x):\n    return x+1"
    # no code block -> raw
    assert "return x" in ev.extract_python("    return x+1", "f")


def test_normalize_and_trivia_match():
    assert ev.normalize_answer("The  U.S.A.!") == "usa"
    assert ev.trivia_match("It was Paris, France.", ["Paris"]) is True
    assert ev.trivia_match("I think it is London", ["Paris", "paris"]) is False


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("all verifier tests passed")
