create:
	python3 -m venv .venv
Simulacion:
	.venv/bin/python Simulacion.py
Metodos:
	.venv/bin/python Metodos.py

install:
	.venv/bin/pip install -r requirements.txt