all: format check
	@echo "Done"
check:
	pyflakes *.py */*.py */*/*.py
format:
	black -l 120 *.py */*.py */*/*.py

.PHONY: check format all
