all: check format
	@echo "Done"
check:
	pyflakes *.py */*.py
format:
	black -l 120 *.py */*.py

.PHONY: check format all
