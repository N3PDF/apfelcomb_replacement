#!/bin/bash
wget -r -np -nH --cut-dirs=1 -l 4  -e robots=off --no-verbose -R index.* https://data.nnpdf.science/pineko/theory_productions/
wget -r -np -nH --cut-dirs=1 -l 4  -e robots=off --no-verbose -P benchmarks -R index.* https://data.nnpdf.science/pineko/data_files/
wget -r -np -nH --cut-dirs=1 -l 4  -e robots=off --no-verbose -P benchmarks -R index.* https://data.nnpdf.science/pineko/fakepdfs/
