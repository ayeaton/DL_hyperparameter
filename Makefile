#!/usr/bin/make -f


SHELL = /bin/bash


setup:
	python3 setup.py sdist bdist_wheel 


