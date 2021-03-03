#!/usr/bin/env bash

nvidia-xconfig -o xcontainer.conf -a
# nvidia-xconfig -o xcontainer.conf -a --use-display-device=None --virtual=1280x1024
X -config xcontainer.conf :88 &
