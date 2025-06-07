all: format check test
	@echo "Done"
check:
	find . -name "*.py" | xargs pyflakes
format:
	find . -name "*.py" | xargs black -l 120

test:
	python -m prompt_control.test_parser

test_graph:
	PYTHONPATH=../../ python -m prompt_control.test_graph

test_encode:
	PYTHONPATH=../../ python -m prompt_control.test_encode

manual_test:
	PYTHONPATH=../../ python -im prompt_control.manual_test

.PHONY: check format all
