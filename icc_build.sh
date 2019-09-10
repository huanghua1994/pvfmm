#./autogen.sh
CXX=icpc MPICXX=mpiicpc CXXFLAGS="-xHost -O3 -qopenmp" ./configure --prefix=$PWD/install --with-fftw-include="$MKLROOT/include/fftw" --with-fftw-lib="-mkl" 
