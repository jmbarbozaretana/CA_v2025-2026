#!/bin/bash

if [ -z "$1" ]; then
    echo "Examen directory '$1' missing"
    exit
fi

RESOLUTION=120
CDIR=$PWD
b=`basename $1`
WDIR=$1
VER=`echo $b | sed -e 's/examen_//g'`
EN=`ls -1 $1 | xargs basename | sed -e "s/examen_\([0-9]*\).tex/\1/" `
EP=E${EN}_P


echo "Working with version $VER"

if [ $? -eq 0 ]; then
    grep input $WDIR/version_*.tex | grep pregunta > tmp_q.tex
    PC=1
    OPTS=(enunciado opA opB opC opD opE opF opG opH)
    
    while read l; do
        q=`echo $l | sed 's/input{\(.*\)}/\1/g'`
        PREFIX=$(printf "%02d" $PC)

        echo "Pregunta $PREFIX: $q"

        SFILE=${EP}${PREFIX}.tex  ## Source file
        PFILE=${EP}${PREFIX}.pdf  ## PDF file

        ## Generate a tex file replacing things in the template
        cat template.tex | sed -e 's@QUESTION@'"$q"'@g'| sed -e 's@DIRECTORY@'"$WDIR"'@1' >$SFILE

        ## Compile twice, as sometimes things need to be rearranged
        pdflatex $SFILE > /dev/null
        pdflatex $SFILE > /dev/null
	 
        ## Extract one image for each page of the pdf.  We assume the first page has
        ## the question and the following pages the options
        pdftoppm -f 1 -png -r $RESOLUTION $PFILE ${EP}${PREFIX}_v${VER}_pag
        mogrify -trim -border 5x5 -bordercolor white ${EP}${PREFIX}_v${VER}_pag*.png


        ## WARNING:
        ## Ubuntu 20.04 uses graphicsmagick instead of imagemagick
        ## and the following lines wouldn't work.
        ## Install imagemagick instead
        
        ## Extract width from the very first page
        WIDTH=`convert ${EP}${PREFIX}_v${VER}_pag-1.png  -print "%w" /dev/null`

        O=0
        for i in `ls -1  ${EP}${PREFIX}_v${VER}_pag*.png | sort`; do
            T=${EP}${PREFIX}_v${VER}_${OPTS[$O]}.png

            echo "$i -> $T"

            HEIGHT=`convert $i -print "%h" /dev/null`

            convert $i -extent ${WIDTH}x${HEIGHT} $T
            rm $i
            
            O=$((O + 1)) 
        done
        
        ## Remove the temporary files
        rm $SFILE
        rm $PFILE
        rm ${EP}${PREFIX}.{aux,log,out}
        
        PC=$((PC + 1))
        
    done<tmp_q.tex
    rm tmp_q.tex
    rm -f *.log
else
    echo "Could not change to '$1'"
fi
