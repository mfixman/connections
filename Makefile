.PHONY: install test lint docs clean

install:
	pip install -e ".[dev,docs]"

test:
	pytest

lint:
	black .
	isort .
	flake8 .
	mypy .

docs:
	cd docs && make clean && make html

clean:
	rm -rf docs/_build/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +
	find . -type f -name .DS_Store -delete 