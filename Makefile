ARGS=
all: format check test
	@echo "Done"

check:
	ty check && ruff check

fix:
	ruff check --fix

format:
	ruff format

test:
	PYTHONPATH=../../ pytest tests/test_parser.py $(ARGS)

test_graph:
	PYTHONPATH=../../  pytest tests/test_graph.py $(ARGS)

test_encode:
	PYTHONPATH=../../  pytest tests/test_encode.py $(ARGS)

test_encode_both:
	TEST_TE="clip_l t5" PYTHONPATH=../../  pytest tests/test_encode.py $(ARGS)

test_heavy: test_graph test_encode_both

manual_test:
	PYTHONPATH=../../ python -im prompt_control.manual_test

.PHONY: check format all
