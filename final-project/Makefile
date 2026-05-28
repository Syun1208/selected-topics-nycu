all: clean build

build:
	@cd bundle && python setup.py bdist_wheel

clean:
	$(RM) -r bundle/dist bundle/build bundle/*.egg-info

.PHONY: build clean