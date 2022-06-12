help:
	cat Makefile

doc: .
	poetry run make -C doc autodoc
	poetry run make -C doc html

tox:
	black -l 110 .
	poetry run tox
