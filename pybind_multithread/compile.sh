g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) single_thread_no_gil.cpp -o single_thread_no_gil$(python3-config --extension-suffix)
g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) single_thread_w_gil.cpp -o single_thread_w_gil$(python3-config --extension-suffix)
g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) multi_thread.cpp -o multi_thread$(python3-config --extension-suffix) --fopenmp
