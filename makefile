
CXX='bout-config --cxx'
CFLAGS='bout-config --cflags'
LD='bout-config --ld'
LDFLAGS='bout-config --libs'

SOURCEC = main.cxx

include $(shell bout-config --config-file)


