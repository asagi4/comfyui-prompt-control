all: format check
	@echo "Done"
check:
	find . -name "*.py" | xargs pyflakes
format:
	find . -name "*.py" | xargs black -l 120

.PHONY: check format all
