hello: hello.cpp
	icpx -fsycl -fsycl-unnamed-lambda -fopenmp hello.cpp -o hello
