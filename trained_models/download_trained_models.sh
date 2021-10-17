#!/bin/bash

wget -v -O models.zip -L \
	http://vision.cs.stonybrook.edu/~shahira/mihc_analysis_dp_paper_resources/models.zip && \
unzip -o models.zip && rm -f models.zip