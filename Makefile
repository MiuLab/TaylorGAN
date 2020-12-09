.DEFAULT_GOAL := all

.PHONY: lint
lint:
	flake8

.PHONY: test-core
test-core:
	TF_CPP_MIN_LOG_LEVEL=3 pytest core/ --cov=core/ --cov-report term-missing

.PHONY: test-library
test-library:
	pytest library/ --cov=library/ --cov-fail-under=20

.PHONY: test-integration
test-integration:
	TF_CPP_MIN_LOG_LEVEL=3 pytest \
	scripts/tests/test_integration.py::TestTrain::test_GAN \
	scripts/tests/test_integration.py::TestEvaluate -vv

.PHONY: test-integration-easy
test-integration-easy:
	TF_CPP_MIN_LOG_LEVEL=3 pytest scripts/tests/test_integration.py::TestTrain::test_GAN

.PHONY: all
all: lint test-library test-core test-integration
