#!/bin/bash

NAVSIM_INSTALLED="/tmp/NAVSIM_INSTALLED"
NAVSIM_DIR="/opt/navsim"
if [ ! -e $NAVSIM_INSTALLED ]; then
    touch $NAVSIM_INSTALLED
    # YOUR_JUST_ONCE_LOGIC_HERE

    cd $NAVSIM_DIR
    ./install-repo.sh

else
    echo "navsim already installed ... skipping."
fi
