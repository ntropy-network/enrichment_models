IMG = "enrichment_models"
GIT_SHA = $(shell git rev-parse --short=10 HEAD)

build_image:
	docker build \
	-t ${IMG}:${GIT_SHA} \
	-t ${IMG}:latest \
	-f ./Dockerfile .

push_image:
	docker push ${IMG}:${GIT_SHA}
	docker push ${IMG}:latest

requirements:
	poetry export -f requirements.txt --without-hashes --output requirements.txt

unittests:
	pytest -s -vv tests/unittests/

integration_tests:
	pytest -s -vv tests/integration/

tests: unittests integration_tests

benchmark:
	python scripts/full_benchmark.py