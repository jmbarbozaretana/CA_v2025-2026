#----------------------------------------------------------------
# project ....: Dissertation
# file .......: Makefile
# authors ....: Pablo Alvarado
# organization: LTI, RWTH Aachen
# creation ...: 24.04.2003
#----------------------------------------------------------------

MAINFILE=$(shell basename $$PWD)

LATEX=/usr/bin/latex -interaction=batchmode -file-line-error-style
PDFLATEX=pdflatex -interaction=batchmode
BIB=biber

# TeX files
TEXFILES = $(wildcard ./*.tex) $(wildcard ./*.sty)
PNGFILES = $(wildcard fig/*.png)
JPGFILES = $(wildcard fig/*.jpg)
FIGFILES = $(wildcard fig/*.fig)
LFIGFILES = $(wildcard fig/*.ltxfig)
PDFFILES = $(patsubst %.eps,%.pdf,$(EPSFILES))
EPNGFILES = $(patsubst %.png,%.eps,$(PNGFILES))
EJPGFILES = $(patsubst %.jpg,%.eps,$(JPGFILES))
EFIGFILES = $(patsubst %.fig,%.eps,$(FIGFILES))
ELFIGFILES = $(patsubst %.ltxfig,%.eps,$(LFIGFILES))
EPSFILES = $(wildcard fig/*.eps) $(EPNGFILES) $(EJPGFILES) $(EFIGFILES) $(ELFIGFILES)

BIBFILES = $(wildcard *.bib)

# implicit rules (pattern rules)

# for eps images from png
fig/%.eps : fig/%.png
	@echo "Converting $< to $@" ; \
	convert -density 100x100 $< 'eps:-' > $@

# for eps images from jpeg
fig/%.eps : fig/%.jpg
	@echo "Converting $< to $@" ; \
	jpeg2ps  $< > $@

# for eps images from fig
# WARNING: fig files may include eps files directly, so, the following trick
#          of changing to fig to go back with each file is done so that fig2dev
#          can find those files.
fig/%.eps : fig/%.fig
	@echo "Converting $< to $@" ; \
	cd fig ; fig2dev -L eps ../$< ../$@

#
# for eps images from ltxfig files (fig files with LaTeX code)
# it is assumed that there exists a file with the same basename but extension
# psfrag.  It must contain the LaTeX code to work with a preliminar eps file
# with extension epstmp.
#
# If the file with extension psfrag does not exist, it will be created with
# a default content, which is: \includegraphics{your_file.epstmp}
#
fig/%.eps : fig/%.ltxfig fig/%.psfrag
	@echo "Converting $< to $@" ; \
	file=`basename $< .ltxfig` ; \
	cd fig ; fig2dev -L eps $$file.ltxfig $$file.epstmp ; \
	if [ ! -f $$file.psfrag ] ; then \
	  echo "\includegraphics{$$file.epstmp}" > $$file.psfrag ; \
	fi ; \
	(echo '\documentclass{minimal}' ; \
	 echo '\usepackage{graphicx,color,amsmath,amssymb,amstext,psfrag}' ; \
	 echo '\begin{document} '; \
	 cat $$file.psfrag ; \
	 echo '\end{document}')> ../$<.tex ; \
	$(LATEX) ../$<.tex ; \
	dvips -Ppdf -o $$file.ps ../$<.dvi ; \
	ps2eps -f $$file.ps ; \
	rm  ../$<.aux ../$<.tex ../$<.log ../$<.dvi $$file.epstmp

fig/%.psfrag:
	file=`basename $@ .psfrag` ; \
	if [ ! -f $@ ] ; then \
	  echo "\includegraphics{$$file.epstmp}" > $@ ; \
	  cp $@ $@.TODO ; \
	fi


# for pdf images from eps
fig/%.pdf : fig/%.eps
	@echo "Converting $< to pdf" ; \
	epstopdf --outfile=$@ $<

# -----------------------------------------------------------------------------
# Targets
# -----------------------------------------------------------------------------

pdf:     $(MAINFILE).pdf

$(MAINFILE).pdf:	$(TEXFILES) $(PDFFILES) $(BIBFILES) config.tex
	@echo "Generating $(MAINFILE).pdf file..."; \
	echo "------------------------------------------" ;\
	echo "Running latex once..." ;\
	$(PDFLATEX) $(MAINFILE) > pdf.log 2>&1 ;\
	if ( $(BIB) $(MAINFILE) >> pdf.log 2>&1 ) ; then \
	  echo " Bibliography ok" ;\
	else \
	  echo " Bibliography failed" ;\
	fi ;\
	if ( $(BIB) $(MAINFILE) >> pdf.log 2>&1 ) ; then \
	  echo " Bibliography ok" ;\
	else \
	  echo " Bibliography failed" ;\
	fi ;\
	$(PDFLATEX) $(MAINFILE) > pdf.log 2>&1 ;\
	latex_count=5 ; \
	while egrep -s 'Rerun (LaTeX|to get cross-references right)' $(MAINFILE).log && [ $$latex_count -gt 0 ] ;\
	    do \
	      echo "Rerunning latex...." ;\
	      $(PDFLATEX) $(MAINFILE).tex >> pdf.log 2>&1 ;\
	      latex_count=`expr $$latex_count - 1` ;\
	    done ;\
	echo "Ready."


config.tex:
	@echo "Use 'make active-completo' o 'make active-solo-a'" ;\
	if ! test -L ./config.tex; then \
		make active-completo; \
	fi

active-completo:
	@makelink=false;\
	if ! test -f ./config.tex ; then \
		makelink=true; \
	fi; \
	if test	-L ./config.tex; then \
		if [ $$(readlink ./config.tex) != "../common/config-both.tex" ]; then \
			makelink=true;\
		fi ;\
	fi;\
	if [ "$$makelink" = "true" ]; then \
		echo "  Activando versión de la Comisión Curricular" ;\
		ln -sfv ../common/config-both.tex ./config.tex ;\
		touch ./config.tex ;\
	fi

completo: active-completo
	@echo "Construyendo PDF completo" ;\
	make pdf ;\

active-solo-a:
	@makelink=false;\
	if ! test -f ./config.tex ; then \
		makelink=true; \
	fi; \
	if test	-L ./config.tex; then \
		if [ $$(readlink ./config.tex) != "../common/config-onlya.tex" ]; then \
			makelink=true;\
		fi ;\
	fi;\
	if [ "$$makelink" = "true" ]; then \
		echo "  Activando versión con solo parte a" ;\
		ln -sfv ../common/config-onlya.tex ./config.tex ;\
		touch ./config.tex ;\
	fi

solo-a: active-solo-a
	@echo "Construyendo PDF con solo parte a" ;\
	make pdf ;\

force-pdf: 
	touch $(MAINFILE).tex ;\
	make pdf

figs:	$(EPSFILES)

pdfigs: figs $(PDFFILES)

partial-clean:
	@echo "Cleaning..." ;\
	rm -f *.log *.lot *.toc *.lof *.aux *.dvi *.glo *.gls ;\
	rm -f *.idx *.ilg *.ind *.bbl *.blg *.brf *.out *.todo *.flc *.xmp ;\
	rm -f *.run.xml *.bcf *.loc *.soc

clean:  partial-clean
	@cd fig ;\
	for e in fig png jpg ltxfig; do \
	  for i in `find . -name "*.$$e"` ; do \
            b=`basename $$i .$$e`.eps ;\
	    if [ -f $$b ] ; then \
	      rm $$b ;\
	    fi \
	  done \
	done ;\
	echo "Done." ;\
	cd ..

clean-all: clean
	@rm -f `find . -name "*~"` \#* fig/*.aux ;\
	rm -fr *.ps *.pdf ;\
	rm -fr `find . -name ".xvpics"` \.#* ;\
	rm -fr fig/*.fig.bak ;\
	echo "All cleaned up"

help:
	@echo "This Makefile provides following targets: " ;\
	echo " ps:        generate $(MAINFILE).ps" ;\
	echo " force-ps:  force generation of $(MAINFILE).ps" ;\
	echo " pdf:       generate $(MAINFILE).pdf" ;\
	echo " force-pdf: force generation of $(MAINFILE).pdf" ;\
	echo " figs:      create eps image files from jpg,png,fig,etc." ;\
	echo " pdfigs:    create pdf image files from jpg,png,fig,etc." ;\
	echo " clean:     remove temporary files but keep final ps and pdf" ;\
	echo " clean-all: like clean but remove also final files"
