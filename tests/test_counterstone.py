import counterstone


def test_version():
    print(dir(counterstone))
    assert counterstone.__version__
