#!/bin/bash

# This script should be executed to make the wavs in a folder
# compatible in the same format.
# The user should provide the path of the folder as the first
# command-line argument.

if [ -n "$1" ]; then
   prefix=$1;
else
    prefix='../wav';
fi
rm -r $prefix/tmpwav
mkdir $prefix/tmpwav
cd $prefix
for f in *.wav; do avconv -i $f $prefix/tmpwav/$f -y; done;
#nikos' line
for f in *.wav; do avconv -i $f -ar 16000 -ac 1 $prefix/tmpwav/$f -y; done;
cd .. 
#-rm -r $prefix
#mv ./tmpwav $prefix
#!/bin/bash

