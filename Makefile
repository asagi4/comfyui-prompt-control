all: format check test
	@echo "Done"

check:
	ty check && ruff check

format:
	ruff format

test:
	python -m prompt_control.test_parser

test_graph:
	PYTHONPATH=../../ python -m prompt_control.test_graph

test_encode:
	PYTHONPATH=../../ python -m prompt_control.test_encode --verbose

test_encode_both:
	TEST_TE="clip_l t5" PYTHONPATH=../../ python -m prompt_control.test_encode --verbose

test_heavy: test_graph test_encode_both

manual_test:
	PYTHONPATH=../../ python -im prompt_control.manual_test

.PHONY: check format all
