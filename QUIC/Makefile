# Author: Matyas A. Sustik
# Send email to sustik@cs.utexas.edu with bug reports and suggestions.

# Acknowledgements
#
# I thank Diane Oyen for the MacOS compile instructions and for
# testing the Makefile.  I thank Bernard Ng for his input regarding
# Windows compile and duality gap.

# You may also compile QUIC using the MEX compiler.  For example:
#
# > mex -llapack QUIC.C QUIC-mex.C -output QUIC.mexa64
#
# Use the proper extension applicable: mex (Octave, Linux), mexglx,
# mexa64 (Matlab, Linux), mexmaci32, mexmaci64 (Matlab, MacOS),
# mexw32, mexw64 (Matlab, Windows).

# Note to Windows users:
#
# It appears that some compilers used on Windows do not recognize the
# .C extension as a C++ source file.  You may mitigate this
# shortcoming by renaming the files using the extension .cpp.

# Notes on problem reports and debugging.
#
# If you find a problem with this Makefile or find a bug in the code
# itself, please send a report to: sustik@cs.utexas.edu.  The
# following will make your report most useful:
# 1. Try to reproduce the problem on Linux and using as small example
#    as you can.
# 2. Save the arguments to the failing function call in a .mat file
#    (version 7) and send it along with your invocation and error
#    description in an email.
#
# Should you decide to attempt to debug the problem yourself then you
# may try the debug version of the program. For example, when working
# under Octave, compile the debug version with:
#
# > make QUIC_g.mex
#
# A gdb debug window should pop up when glassofast_g is
# called. Remember to switch back to the optimized version once
# finished and please report the bug and the fix if you have one.

# Contact your administrator if you do not know where your matlab is
# installed.
MATLAB_ROOT = /lusr/share/software/matlab-2011a
# Normally you would not need to modify lines bellow this one.  If you
# need to, please let us know how we can improve this Makefile.

MATLAB_INCLUDE = -I$(MATLAB_ROOT)/extern/include
OCTAVE_INCLUDE = -I/usr/include/octave
# Octave has a nice feature to tell us the library locations:
OCTAVE_LIBS = $(shell mkoctfile -p LFLAGS)

CXX=g++

CXXFLAGS = -Wall -fpic -pthread -shared  -fno-omit-frame-pointer -ansi -D_GNU_SOURCE -D_FILE_OFFSET_BITS=64
CXXOPTFLAGS = -O3 -DNDEBUG $(CXXFLAGS)
CXXDBGFLAGS = -g -DNDEBUG -DGDEBUG $(CXXFLAGS)

LDFLAGS = -Wall -pthread -shared -lmwblas -lmwlapack -lmx -lmex -lmat -lm -lgfortran

# Linker flags for Matlab on Linux:
LDFLAGS_32 = -m32 -Wl,--version-script,$(MATLAB_ROOT)/extern/lib/glnx86/mexFunction.map -Wl,-rpath-link,$(MATLAB_ROOT)/bin/glnx86 -L$(MATLAB_ROOT)/bin/glnx86 $(LDFLAGS)
LDFLAGS_64 = -m64 -Wl,--version-script,$(MATLAB_ROOT)/extern/lib/glnxa64/mexFunction.map -Wl,-rpath-link,$(MATLAB_ROOT)/bin/glnxa64 -L$(MATLAB_ROOT)/bin/glnxa64 $(LDFLAGS)

# Linker flags for Matlab on MacOS:
LDMACFLAGS_32 = -m32 -Wl,-exported_symbols_list,$(MATLAB_ROOT)/extern/lib/mexmaci32/mexFunction.map -Wl,-rpath,$(MATLAB_ROOT)/bin/maci32 -L$(MATLAB_ROOT)/bin/maci32 $(LDFLAGS)
LDMACFLAGS_64 = -m64 -Wl,-exported_symbols_list,$(MATLAB_ROOT)/extern/lib/mexmaci64/mexFunction.map -Wl,-rpath,$(MATLAB_ROOT)/bin/maci64 -L$(MATLAB_ROOT)/bin/maci64 $(LDFLAGS)

# Linker flags for Octave on Linux:
LDOCTMEXFLAGS = -shared -Wl,-Bsymbolic -loctave -lcruft -llapack -lblas -lfftw3 -lfftw3f -lreadline -lncurses -ldl -lhdf5 -lz -lm $(OCTAVE_LIBS)

.SUFFIXES:

# C++ rules for MEX wrappers:
%-mex_32_g.o: %-mex.C
	$(CXX) $(CXXDBGFLAGS) $(MATLAB_INCLUDE) -m32 -c $< -o $@

%-mex_32.o: %-mex.C
	$(CXX) $(CXXOPTFLAGS) $(MATLAB_INCLUDE) -m32 -c $< -o $@

%-mex_64_g.o: %-mex.C
	$(CXX) $(CXXDBGFLAGS) $(MATLAB_INCLUDE) -m64 -c $< -o $@

%-mex_64.o: %-mex.C
	$(CXX) $(CXXOPTFLAGS) $(MATLAB_INCLUDE) -m64 -c $< -o $@

%-oct_g.o: %-mex.C
	$(CXX) $(CXXDBGFLAGS) $(OCTAVE_INCLUDE) -c $< -o $@

%-oct.o: %-mex.C
	$(CXX) $(CXXOPTFLAGS) $(OCTAVE_INCLUDE) -c $< -o $@

# C++ rules for algorithm programs:
%_32_g.o: %.C
	$(CXX) $(CXXDBGFLAGS) $(MATLAB_INCLUDE) -m32 -c $< -o $@

%_32.o: %.C
	$(CXX) $(CXXOPTFLAGS) $(MATLAB_INCLUDE) -m32 -c $< -o $@

%_64_g.o: %.C
	$(CXX) $(CXXDBGFLAGS) $(MATLAB_INCLUDE) -m64 -c $< -o $@

%_64.o: %.C
	$(CXX) $(CXXOPTFLAGS) $(MATLAB_INCLUDE) -m64 -c $< -o $@

%_g.o: %.C
	$(CXX) $(CXXDBGFLAGS) $(OCTAVE_INCLUDE) -c $< -o $@

%.o: %.C
	$(CXX) $(CXXOPTFLAGS) $(OCTAVE_INCLUDE) -c $< -o $@

# Link Matlab executables:
%_g.mexa64 : %-mex_64_g.o %_64_g.o
	$(CXX) $(LDFLAGS_64) $^ -o $@

%.mexa64 : %-mex_64.o %_64.o
	$(CXX) $(LDFLAGS_64) $^ -o $@

%_g.mexglx : %-mex_32_g.o %_32_g.o
	$(CXX) $(LDFLAGS_32) $^ -o $@

%.mexglx : %-mex_32.o %_32.o
	$(CXX) $(LDFLAGS_32) $^ -o $@

%_g.mexmaci64 : %-mex_64_g.o %_64_g.o
	$(CXX) $(LDMACFLAGS_64) $^ -o $@

%.mexmaci64 : %-mex_64.o %_64.o
	$(CXX) $(LDMACFLAGS) $^ -o $@

%_g.mexmaci32 : %-mex_32_g.o %_32_g.o
	$(CXX) $(LDMACFLAGS_32) $^ -o $@

%.mexmaci32 : %-mex_32.o %_32.o
	$(CXX) $(LDMACFLAGS_32) $^ -o $@

# Link Octave executables:
%_g.mex : %-oct_g.o %_g.o
	g++ $(LDOCTMEXFLAGS) $^ -o $@

%.mex : %-oct.o %.o
	g++ $(LDOCTMEXFLAGS) $^ -o $@ 

.SECONDARY :

clean :
	rm -f *.o *.mexa64 *.mexglx *.oct *.mex

