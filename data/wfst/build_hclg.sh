#!/bin/bash
# WFST HCLG Graph Construction
# Requires OpenFST and Kaldi tools

set -e

# Compile FSTs
fstcompile --acceptor=false L.fst.txt L.fst
fstcompile --acceptor=false G.fst.txt G.fst
fstcompile --acceptor=true C.fst.txt C.fst
fstcompile --acceptor=true T.fst.txt T.fst

# Compose HCLG = H CL G
# H = input symbols (pdfids)
# C = context FST
# L = lexicon FST
# G = language model FST

# This is a simplified version
# In production, use Kaldi's make-hclg.sh
