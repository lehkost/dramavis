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

# delete old superposter
if [ -e superposter.png ]
then
   rm superposter.png
fi

# build string containing all plays in chronological order
for n in `ls [0-9]*.png | sort`
do
	filelist=$filelist'"'$n'" '
done

# hand over filelist to ImageMagick's montage feature
montage -monitor -tile 26x18 -geometry 1200x1200+0+0 $filelist superposter.png

# blank line
echo
