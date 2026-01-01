  nvcc -std=c++17 -arch=sm_100 \
       -I../../../include \
       atom_concept_simple.cu \
       -o atom_concept_simple
