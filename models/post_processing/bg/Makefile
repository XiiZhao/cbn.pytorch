CXXFLAGS = -I /home/xxxx/xxx/OpenCV/include -I clipper/cpp/libs/include/polyclipping -I /usr/local/anaconda3/include/python3.6m -I pybind11/include -std=c++11 -O3

#DEPS = $(shell find include -xtype f)
CXX_SOURCES = boundary_guided.cpp
OPENCV = `pkg-config --cflags --libs opencv`
CLIPPING = `pkg-config --cflags --libs polyclipping`
LIB_SO = bg.so
$(LIB_SO): $(CXX_SOURCES) $(DEPS)
	$(CXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $(CXX_SOURCES) --shared -fPIC $(OPENCV) $(CLIPPING) 
clean:
	rm -rf $(LIB_SO)
