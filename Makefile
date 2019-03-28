#!/usr/bin/make -f


SHELL = /bin/sh


setup:
	python3 setup.py sdist bdist_wheel
