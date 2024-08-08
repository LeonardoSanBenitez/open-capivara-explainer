.ONESHELL: # Source: https://stackoverflow.com/a/30590240

auth:
	echo "Login not yet configured"

run-local: auth
	echo "go to http://localhost:8888/tree?token=5074880b-def3-4506-be31-0f60c98cc42b"
	if docker-compose ps | grep explainer-notebooks >/dev/null; then \
		echo "Container is already running."; \
	else \
		echo "Container is not running. Starting it now..."; \
		docker-compose down; \
		docker-compose up -d; \
	fi;

stop-local:
	docker-compose down

test: run-local
	mypy --install-types --non-interactive

	echo "\n\n-------\nMypy checks\n-------"
	docker-compose exec notebooks mypy ./libs --no-warn-incomplete-stub --disable-error-code import-untyped --explicit-package-bases

	echo "\n\n-------\nPycodestyle checks\n-------"
	docker-compose exec notebooks pycodestyle --exclude='.venv,docs,.runs' --max-line-length=200 --ignore='E121,E123,E126,E226,E24,E251,E704,W503,W504,E225,E226,E252,W605,E721,E731' ./libs
	
	echo "\n\n-------\nPytest checks\n-------"
	docker-compose exec notebooks python3 -m pytest tests
