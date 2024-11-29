#define M_PI 3.14159265358979323846
#include <vector>
#include <cmath>
#include <algorithm>

std::vector<unsigned char> grayscale_image(unsigned char* img, int height, int width, int channels) {
    std::vector<unsigned char> grayscale(width * height);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int idx = (i * width + j) * channels;
            unsigned char r = img[idx];
            unsigned char g = img[idx + 1];
            unsigned char b = img[idx + 2];
            grayscale[i * width + j] = static_cast<unsigned char>(r * 0.299 + g * 0.587 + b * 0.114); // RGB to grayscale
        }
    }
    return grayscale;
}

std::vector<std::vector<float>> generate_gaussian_kernel(int kernel_size, float sigma) {
    int half_size = kernel_size / 2;
    std::vector<std::vector<float>> kernel(kernel_size, std::vector<float>(kernel_size));
    float sum = 0.0;

    for (int i = -half_size; i <= half_size; i++) {
        for (int j = -half_size; j <= half_size; j++) {
            float value = std::exp(-(i * i + j * j) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
            kernel[i + half_size][j + half_size] = value;
            sum += value;
        }
    }

    // Normalize the kernel so that the sum of all values is 1
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            kernel[i][j] /= sum;
        }
    }

    return kernel;
}

// Function to apply Gaussian blur
std::vector<unsigned char> gaussian_blur(const std::vector<unsigned char>& grayscale, int height, int width, int kernel_size, float sigma) {
    int half_size = kernel_size / 2;
    std::vector<std::vector<float>> kernel = generate_gaussian_kernel(kernel_size, sigma);
    std::vector<unsigned char> blurred_image(height * width);

    // Apply convolution
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float pixel_value = 0.0;

            for (int ki = -half_size; ki <= half_size; ki++) {
                for (int kj = -half_size; kj <= half_size; kj++) {
                    int ni = i + ki;
                    int nj = j + kj;

                    // Check for boundary conditions
                    if (ni >= 0 && ni < height && nj >= 0 && nj < width) {
                        pixel_value += grayscale[ni * width + nj] * kernel[ki + half_size][kj + half_size];
                    }
                }
            }

            blurred_image[i * width + j] = static_cast<unsigned char>(pixel_value);
        }
    }

    return blurred_image;
}

std::pair<std::vector<float>, std::vector<float>> calculateIntensityGradients(
    const std::vector<unsigned char>& blurredImage, int height, int width) {

    // Sobel kernels
    int Gx[5][5] = {
        {-1,  0,  1,  0, -1},
        {-2,  0,  2,  0, -2},
        {-3,  0,  3,  0, -3},
        {-2,  0,  2,  0, -2},
        {-1,  0,  1,  0, -1}
    };
    int Gy[5][5] = {
        {-1, -2, -3, -2, -1},
        { 0,  0,  0,  0,  0},
        { 1,  2,  3,  2,  1},
        { 0,  0,  0,  0,  0},
        {-1, -2, -3, -2, -1}
    };

    // Output gradient magnitude and direction arrays
    std::vector<float> gradientMagnitude(height * width, 0);
    std::vector<float> gradientDirection(height * width, 0);

    // Loop over the image, skipping the edges
    for (int i = 1; i < height - 1; ++i) {
        for (int j = 1; j < width - 1; ++j) {
            float gx = 0, gy = 0;

            // Apply Sobel kernels
            for (int ki = -1; ki <= 1; ++ki) {
                for (int kj = -1; kj <= 1; ++kj) {
                    int pixel = blurredImage[(i + ki) * width + (j + kj)];
                    gx += pixel * Gx[ki + 1][kj + 1];
                    gy += pixel * Gy[ki + 1][kj + 1];
                }
            }

            // Calculate magnitude and direction
            int idx = i * width + j;
            gradientMagnitude[idx] = std::sqrt(gx * gx + gy * gy);
            gradientDirection[idx] = std::atan2(gy, gx); // Angle in radians
        }
    }

    return { gradientMagnitude, gradientDirection };
}

std::vector<unsigned char> nonMaximumSuppression(
    const std::vector<float>& gradientMagnitude,
    const std::vector<float>& gradientDirection,
    int height, int width) {

    std::vector<unsigned char> suppressedImage(height * width, 0);

    for (int i = 1; i < height - 1; ++i) {
        for (int j = 1; j < width - 1; ++j) {
            int idx = i * width + j;
            float angle = gradientDirection[idx] * 180.0 / M_PI; // Convert to degrees
            angle = (angle < 0) ? angle + 180 : angle; // Normalize to [0, 180)

            float mag = gradientMagnitude[idx];
            float mag1 = 0, mag2 = 0;

            // Determine neighboring pixels to compare
            if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180)) {
                mag1 = gradientMagnitude[idx + 1];     // Right
                mag2 = gradientMagnitude[idx - 1];     // Left
            }
            else if (angle >= 22.5 && angle < 67.5) {
                mag1 = gradientMagnitude[(i - 1) * width + (j + 1)]; // Top-right
                mag2 = gradientMagnitude[(i + 1) * width + (j - 1)]; // Bottom-left
            }
            else if (angle >= 67.5 && angle < 112.5) {
                mag1 = gradientMagnitude[(i - 1) * width + j]; // Top
                mag2 = gradientMagnitude[(i + 1) * width + j]; // Bottom
            }
            else if (angle >= 112.5 && angle < 157.5) {
                mag1 = gradientMagnitude[(i - 1) * width + (j - 1)]; // Top-left
                mag2 = gradientMagnitude[(i + 1) * width + (j + 1)]; // Bottom-right
            }

            // Suppress non-maximum gradients
            if (mag >= mag1 && mag >= mag2) {
                suppressedImage[idx] = static_cast<unsigned char>(mag);
            }
            else {
                suppressedImage[idx] = 0;
            }
        }
    }

    return suppressedImage;
}

std::vector<unsigned char> doubleThresholding(
    const std::vector<unsigned char>& edges,
    int height, int width,
    int lowThreshold, int highThreshold) {

    std::vector<unsigned char> thresholdedImage(height * width, 0);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int idx = i * width + j;
            int pixelValue = edges[idx];

            // Strong edge
            if (pixelValue >= highThreshold) {
                thresholdedImage[idx] = 255; // Strong edge
            }
            // Weak edge
            else if (pixelValue >= lowThreshold) {
                thresholdedImage[idx] = 128; // Weak edge
            }
            // Non-edge (already 0 by default)
        }
    }

    return thresholdedImage;
}

void edgeTrackingByHysteresis(
    std::vector<unsigned char>& thresholdedImage,
    int height, int width) {

    // Directions to check the 8 neighbors (N, NE, E, SE, S, SW, W, NW)
    const int directions[8][2] = { {-1, 0}, {1, 0}, {0, -1}, {0, 1},
                                   {-1, -1}, {-1, 1}, {1, -1}, {1, 1} };

    for (int i = 1; i < height - 1; ++i) {
        for (int j = 1; j < width - 1; ++j) {
            int idx = i * width + j;

            // If the pixel is a weak edge
            if (thresholdedImage[idx] == 128) {
                // Check if it is connected to a strong edge
                bool connectedToStrong = false;

                for (int d = 0; d < 8; ++d) {
                    int ni = i + directions[d][0];
                    int nj = j + directions[d][1];
                    int nidx = ni * width + nj;

                    if (thresholdedImage[nidx] == 255) {
                        connectedToStrong = true;
                        break;
                    }
                }

                // If connected to a strong edge, keep it as an edge
                if (connectedToStrong) {
                    thresholdedImage[idx] = 255;
                }
                // Otherwise, set it to 0 (non-edge)
                else {
                    thresholdedImage[idx] = 0;
                }
            }
        }
    }
}

float otsuThreshold(const std::vector<unsigned char>& grayscale, int height, int width) {
    // Step 1: Compute the histogram
    std::vector<int> histogram(256, 0);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int pixelVal = grayscale[i * width + j];
            histogram[pixelVal]++;
        }
    }

    // Step 2: Calculate total number of pixels
    int totalPixels = height * width;

    // Step 3: Compute cumulative sum and cumulative mean
    std::vector<float> cumulativeSum(256, 0.0f);
    std::vector<float> cumulativeMean(256, 0.0f);
    cumulativeSum[0] = histogram[0];
    cumulativeMean[0] = 0.0f;

    for (int i = 1; i < 256; i++) {
        cumulativeSum[i] = cumulativeSum[i - 1] + histogram[i];
        cumulativeMean[i] = cumulativeMean[i - 1] + i * histogram[i];
    }

    // Step 4: Compute global mean
    float globalMean = cumulativeMean[255] / totalPixels;

    // Step 5: Find the threshold that maximizes between-class variance
    float maxVariance = 0.0f;
    int optimalThreshold = 0;

    for (int t = 0; t < 256; t++) {
        if (cumulativeSum[t] == 0 || cumulativeSum[t] == totalPixels) {
            continue;  // Skip if the class has no pixels
        }

        float weightBackground = cumulativeSum[t] / totalPixels;
        float weightForeground = 1.0f - weightBackground;

        float meanBackground = cumulativeMean[t] / cumulativeSum[t];
        float meanForeground = (cumulativeMean[255] - cumulativeMean[t]) / (totalPixels - cumulativeSum[t]);

        // Compute between-class variance
        float betweenClassVariance = weightBackground * weightForeground * (meanBackground - meanForeground) * (meanBackground - meanForeground);

        // Update the optimal threshold if the between-class variance is maximized
        if (betweenClassVariance > maxVariance) {
            maxVariance = betweenClassVariance;
            optimalThreshold = t;
        }
    }

    return optimalThreshold;
}

void applyOtsuThresholding(const std::vector<unsigned char>& grayscale, int height, int width, std::vector<unsigned char>& output) {
    // Find the optimal threshold using Otsu's method
    float threshold = otsuThreshold(grayscale, height, width);

    // Apply the threshold to the grayscale image
    output.resize(height * width);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int pixelVal = grayscale[i * width + j];
            // If the pixel value is greater than the threshold, set it to 255 (white)
            // Otherwise, set it to 0 (black)
            output[i * width + j] = (pixelVal > threshold) ? 255 : 0;
        }
    }
}

void adaptiveThresholding(const std::vector<unsigned char>& grayscale, int height, int width, std::vector<unsigned char>& output) {
    int blockSize = 15;  // Size of the block used to calculate the threshold
    int C = 10;  // Constant subtracted from the mean or weighted sum

    output.resize(height * width);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int sum = 0, count = 0;
            // Calculate local neighborhood sum
            for (int bi = -blockSize / 2; bi <= blockSize / 2; bi++) {
                for (int bj = -blockSize / 2; bj <= blockSize / 2; bj++) {
                    int ni = i + bi;
                    int nj = j + bj;
                    if (ni >= 0 && ni < height && nj >= 0 && nj < width) {
                        sum += grayscale[ni * width + nj];
                        count++;
                    }
                }
            }
            int mean = sum / count;
            output[i * width + j] = (grayscale[i * width + j] > mean - C) ? 255 : 0;
        }
    }
}