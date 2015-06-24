# this bash script generates a dramavis superposter
# composed of all PNG files in this directory
# using ImageMagick
#
# examples:
# montage -tile 10x70 -geometry 40x40+1+1 *.png superposter.png
# montage -tile 32x21 -geometry 300x300+1+1 *.png superposter.png
#
# explanations:
# http://www.csb.yale.edu/userguides/image/im/www/montage.html
# http://www.imagemagick.org/Usage/montage/#geometry_size

if [ -e superposter.png ]
then
   rm superposter.png
fi

for n in `ls BC*.png | sort -r`
do
	filelist=$filelist$n' '
done

for n in `ls [0-9]*.png | sort`
do
	filelist=$filelist$n' '
done

montage -monitor -tile 10x70 -geometry 100x100+1+1 $filelist superposter.png

echo
