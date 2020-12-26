.DEFAULT_GOAL := all

.PHONY: lint
lint:
	flake8

.PHONY: test
test:
	TF_CPP_MIN_LOG_LEVEL=3 pytest src/ --ignore src/scripts --cov=src/ --cov-fail-under=20 --cov-report term-missing

.PHONY: test-integration
test-integration:
	pytest src/scripts/tests/test_integration.py

.PHONY: test-integration-info
test-integration-info:
	TF_CPP_MIN_LOG_LEVEL=3 pytest src/scripts/tests/test_integration.py::TestTrain::test_GAN \
	src/scripts/tests/test_integration.py::TestSaveLoad \
	src/scripts/tests/test_integration.py::TestEvaluate -vv

.PHONY: test-integration-easy
test-integration-easy:
	TF_CPP_MIN_LOG_LEVEL=3 pytest src/scripts/tests/test_integration.py::TestTrain::test_GAN

.PHONY: all
all: lint test test-integration

.PHONY: clean
clean:
	rm -rf `find . -name __pycache__`
	rm -f `find . -type f -name '*.py[co]' `
	rm -f `find . -type f -name '*~' `
	rm -f `find . -type f -name '.*~' `
	rm -rf .cache
	rm -rf htmlcov
	rm -rf *.egg-info
	rm -f .coverage
	rm -f .coverage.*
	rm -rf build
	rm -rf dist
	python setup.py clean
