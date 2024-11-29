#include <canny_edge_detector.cu>

void launch_canny_edge_detection_kernel(unsigned char* img, unsigned char* output, int width, int height, int kernel_size, float* kernel);