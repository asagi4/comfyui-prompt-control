import logging


def pytest_runtest_setup(item):
    logging.getLogger("comfyui-prompt-control").setLevel(logging.CRITICAL)
