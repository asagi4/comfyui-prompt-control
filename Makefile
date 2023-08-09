all: check format
	@echo "Done"
check:
	pyflakes **/*.py
format:
	black -l 120 **/*.py

.PHONY: check format all
