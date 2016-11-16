#!/bin/bash

# Appropriate variables describing the dataset paths
dataset='../../../../Dataset/';
path1='./complete_dataset/';

<<<<<<< HEAD
# The first argument is the path where the dataset consisting of the WAV spectrograms will
# be stored.

# The second argument is the path where the WAVs have been distributed after the parsing
# of the .json database file.

# The third argument is the source path of the folder in which both the converted WAVs are
# after makeCompatibleWavs script has been called and the spectrograms after generateSpectrograms
# script are stored.
dataset=$1
path1=$2
path2=$3

#path2='../../'
literalNegativeClass='not_'

# Here are the suffices of the dataset images after augmentation
suffix1='.png'
#suffix2='_02A.jpg'
#suffix3='_02B.jpg'
suffix2='_rnoise01.png'
suffix3='_rnoise02.png'
suffix4='_rnoise03.png'
suffix5='_rnoise11.png'
suffix6='_rnoise12.png'
suffix7='_rnoise13.png'
suffix8='_rnoise21.png'
suffix9='_rnoise22.png'
suffix10='_rnoise23.png'
=======
# The first argument is the path where all the images after augmentation are stored
# The second argument is the path where the dataset will be distributed in class folders
dataset=$1
path1=$2

#path2='../../'
#literalNegativeClass='not_'

# Here are the suffices of the dataset images after augmentation
suffix1='.jpg'
suffix2='_02A.jpg'
suffix3='_02B.jpg'
>>>>>>> a2a9eb7d152576e80ef84551219b8a4d10d2c4d4

echo ">>> This is the script for distributing the dataset images (after augmentation)"
echo "    to their respective class folder"

echo

DONE=false
until $DONE; do
	printf ">>> Please give the name of each class you want its images to be distributed\n    to their folders. To exit press Ctlr+D: "
	
	read || DONE=true
	if [ ! $REPLY = EOF ]; then
		echo
		echo ">>> Images belonging to class $REPLY were distributed successfully"
		echo

<<<<<<< HEAD
		prefixPositiveClass=$path1/$REPLY/$REPLY

		#echo $prefixPositiveClass
		#echo $path1/$REPLY
		
		classFolderPath = $dataset/$REPLY

		if [ ! -d "$classFolderPath" ]; then
			mkdir -p $dataset/$REPLY
			#echo $dataset/$REPLY
		fi
		
		positive_classFolderPath = $dataset/$REPLY/$REPLY

		if [ ! -d "$positive_classFolderPath" ]; then
			mkdir -p $dataset/$REPLY/$REPLY
			#echo $dataset/$REPLY/$REPLY
		fi

		cd $prefixPositiveClass
		
		for f in *.wav; do
			pathToEachFile = $path2/$f1
			f1=${f%.wav}
			png1=$suffix1;
			png2=$pathToEachFile$suffix2;
			png3=$pathToEachFile$suffix3;
			png4=$pathToEachFile$suffix4;
			png5=$pathToEachFile$suffix5;
			png6=$pathToEachFile$suffix6;
			png7=$pathToEachFile$suffix7;
			png8=$pathToEachFile$suffix8;
			png9=$pathToEachFile$suffix9;
			png10=$pathToEachFile$suffix10;
			#echo $png1
			#echo $f$suffix1
			#cp $png1 $dataset/$REPLY/$REPLY/$f$suffix1
			#cp $png2 $dataset/$REPLY/$REPLY/$f$suffix2
			#cp $png3 $dataset/$REPLY/$REPLY/$f$suffix3
			#cp $png4 $dataset/$REPLY/$REPLY/$f$suffix4
			#cp $png5 $dataset/$REPLY/$REPLY/$f$suffix5
			#cp $png6 $dataset/$REPLY/$REPLY/$f$suffix6
			#cp $png7 $dataset/$REPLY/$REPLY/$f$suffix7
			#cp $png8 $dataset/$REPLY/$REPLY/$f$suffix8
			#cp $png9 $dataset/$REPLY/$REPLY/$f$suffix9
			#cp $png10 $dataset/$REPLY/$REPLY/$f$suffix10
=======
		prefixPositiveClass=$path1$REPLY/$REPLY

		if [ ! -d "$$path1$REPLY" ]; then
			mkdir -p $$path1$REPLY
		fi

		if [ ! -d "$prefixPositiveClass" ]; then
			mkdir -p $prefixPositiveClass
		fi

		cd $prefixPositiveClass
		for f in *.wav; do
			f1=${f%.wav}
			jpg1=$dataset$f1$suffix1;
			jpg2=$dataset$f1$suffix2;
			jpg3=$dataset$f1$suffix3;
			echo $jpg1
			echo $jpg2
			echo $jpg3
			#cp $jpg1 $f$suffix1
			#cp $jpg2 $f$suffix2
			#cp $jpg3 $f$suffix3
>>>>>>> a2a9eb7d152576e80ef84551219b8a4d10d2c4d4
		done

		prefixNegativeClass=$path2$REPLY/$literalNegativeClass$REPLY

<<<<<<< HEAD
		if [ ! -d "$dataset$prefixNegativeClass" ]; then
			mkdir -p $dataset/$REPLY/$literalNegativeClass$REPLY
		fi
		
		cd $path1/$REPLY/$literalNegativeClass$REPLY
		for f in *.wav; do
			f1=${f%.wav}
			png2=$pathToEachFile$suffix2;
			png3=$pathToEachFile$suffix3;
			png4=$pathToEachFile$suffix4;
			png5=$pathToEachFile$suffix5;
			png6=$pathToEachFile$suffix6;
			png7=$pathToEachFile$suffix7;
			png8=$pathToEachFile$suffix8;
			png9=$pathToEachFile$suffix9;
			png10=$pathToEachFile$suffix10;
			echo $png1
			echo $f$suffix1
			#cp $png1 $dataset/$REPLY/$literalNegativeClass$REPLY/$f$suffix1
			#cp $png2 $dataset/$REPLY/$literalNegativeClass$REPLY/$f$suffix2
			#cp $png3 $dataset/$REPLY/$literalNegativeClass$REPLY/$f$suffix3
			#cp $png4 $dataset/$REPLY/$literalNegativeClass$REPLY/$f$suffix4
			#cp $png5 $dataset/$REPLY/$literalNegativeClass$REPLY/$f$suffix5
			#cp $png6 $dataset/$REPLY/$literalNegativeClass$REPLY/$f$suffix6
			#cp $png7 $dataset/$REPLY/$literalNegativeClass$REPLY/$f$suffix7
			#cp $png8 $dataset/$REPLY/$literalNegativeClass$REPLY/$f$suffix8
			#cp $png9 $dataset/$REPLY/$literalNegativeClass$REPLY/$f$suffix9
			#cp $png10 $dataset/$REPLY/$literalNegativeClass$REPLY/$f$suffix10
=======
		if [ ! -d "$prefixNegativeClass" ]; then
			mkdir -p $prefixNegativeClass
		fi
		
		cd $prefixNegativeClass
		for f in *.wav; do
			f1=${f%.wav}
			jpg1=$dataset$f1$suffix1;
			jpg2=$dataset$f1$suffix2;
			jpg3=$dataset$f1$suffix3;
			echo $jpg1
			echo $jpg2
			echo $jpg3
			#cp $jpg1 $f$suffix1
			#cp $jpg2 $f$suffix2
			#cp $jpg3 $f$suffix3
>>>>>>> a2a9eb7d152576e80ef84551219b8a4d10d2c4d4
		done
	fi
done
echo

#!/bin/bash
