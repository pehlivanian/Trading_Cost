FILE=Simple_Cost

all: $(FILE).pdf

.PHONY: clean
clean:
	\rm -f *.aux *.blg *.bbl *.log *.out *.pdf

$(FILE).pdf: $(FILE).tex
	pdflatex $(FILE)
