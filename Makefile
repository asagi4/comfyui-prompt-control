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

.PHONY: check format all
