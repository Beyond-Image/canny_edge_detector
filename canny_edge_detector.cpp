#include <iostream>
#include <stdlib.h>
#include <cstdio>
#include <string>
#include <canny_regular_functions.h>
#include <gpu_kernels.h>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main(int argc, char** argv) {
	int height, width, channels;
	unsigned char* img = stbi_load(argv[1], &width, &height, &channels, 0);
	int kernel_size = std::stoi(argv[2]);
	if (img == NULL) {
		std::cout << "Error loading image: " << stbi_failure_reason() << std::endl;
		return -1;
	}
	std::cout << "Loaded image: " << std::endl;
	std::cout << "Image Width: "  << width << std::endl;
	std::cout << "Image Height: " << height<< std::endl;
	std::cout << "Image Channels: " << channels << std::endl;

	if (argv[3] != "cuda") {
		int deviceCount;
		cudaGetDeviceCount(&deviceCount);

		for (int deviceIndex = 0; deviceIndex < deviceCount; ++deviceIndex) {
			cudaDeviceProp deviceProp;
			cudaGetDeviceProperties(&deviceProp, deviceIndex);

			std::cout << "Device " << deviceIndex << ": " << deviceProp.name << std::endl;
			std::cout << "  Total Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
			std::cout << "  Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
		}

		return 0;
	}
	else {
		std::cerr << "Canny Edge Detection Through CPU" << std::endl;
		std::vector<unsigned char> gray_image = grayscale_image(img, width, height, channels);
		stbi_write_png("grayscale.png", width, height, 1, gray_image.data(), width);
		float sigma = 1.0;
		std::vector<unsigned char> blurred_image = gaussian_blur(gray_image, height, width, kernel_size, sigma);
		stbi_write_png("blurred_image.png", width, height, 1, blurred_image.data(), width);

		auto [gradientMagnitude, gradientDirection] = calculateIntensityGradients(blurred_image, height, width);
		std::vector<unsigned char> edges = nonMaximumSuppression(gradientMagnitude, gradientDirection, height, width);
		//int lowThreshold = 50; int highThreshold = 150;
		//std::vector<unsigned char> thresholdedEdges = doubleThresholding(edges, height, width, lowThreshold, highThreshold);
		//edgeTrackingByHysteresis(thresholdedEdges, height, width);

		std::vector<unsigned char> finalEdgeMap;
		applyOtsuThresholding(edges, height, width, finalEdgeMap);
		/*adaptiveThresholding(edges, height, width, finalEdgeMap);
		for (size_t i = 0; i < finalEdgeMap.size(); i++) {
			finalEdgeMap[i] = 255 - finalEdgeMap[i];
		}*/


		stbi_write_png("edge.png", width, height, 1, finalEdgeMap.data(), width);
	}


	return 0;
}

