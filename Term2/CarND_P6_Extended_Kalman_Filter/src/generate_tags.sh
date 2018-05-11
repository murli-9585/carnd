#!/bin/bash
# Generate tag files.
# Usage: copy this to one of the branches, run it before actually using the
# cscope and ctags.
set -x
find $PWD/ -name '*.cpp' > cscope_unsrt.files
find $PWD/ -name '*.py' >> cscope_unsrt.files
find $PWD/ -name '*.c' >> cscope_unsrt.files
find $PWD/ -name '*.h' >> cscope_unsrt.files
find $PWD/ -name '*.y' >> cscope_unsrt.files
find $PWD/ -name '*.l' >> cscope_unsrt.files
find $PWD/ -name '*.perl' >> cscope_unsrt.files
find $PWD/ -name '*.pl' >> cscope_unsrt.files
sort cscope_unsrt.files > cscope.files
rm -f cscope_unsrt.files

# Run cscope to generate DB.
cscope -Rb
ctags -L cscope.files

# Set CSCOPE_DB so files can be searched from anywhere.
CSCOPE_DB= $PWD/cscope.out
export CSCOPE_DB
