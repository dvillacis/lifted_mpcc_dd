#!/bin/bash
# Kodak (train kodim01-16, held out 17-24): ./run_kodak.sh SIZE [tv_learn options].
# Same as ./run_dataset.sh kodak SIZE ...; see there for NP, NSUB, TAG, FORCE.
exec "$(dirname "$0")/run_dataset.sh" kodak "$@"
