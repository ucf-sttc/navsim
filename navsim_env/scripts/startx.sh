#!/usr/bin/env bash

nvidia-xconfig -o xcontainer.conf -a
X -config xcontainer.conf :88 &
