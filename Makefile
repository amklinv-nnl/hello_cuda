hello: hello.cu
	nvcc -Xcompiler "-fopenmp" hello.cu -o hello
