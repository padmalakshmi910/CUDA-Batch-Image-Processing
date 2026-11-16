#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#define CUDA_CHECK(call) do { \
  cudaError_t err = (call); \
  if (err != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << "\n"; \
    return 1; \
  } \
} while (0)

struct Image {
  int width = 0, height = 0;
  std::vector<unsigned char> pixels;
  fs::path path;
};

bool ReadPpm(const fs::path& path, Image* image) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  std::string magic;
  int max_value = 0;
  in >> magic >> image->width >> image->height >> max_value;
  in.get();
  if (magic != "P6" || max_value != 255 ||
      image->width <= 0 || image->height <= 0) return false;
  size_t bytes = static_cast<size_t>(image->width) * image->height * 3;
  image->pixels.resize(bytes);
  in.read(reinterpret_cast<char*>(image->pixels.data()), bytes);
  image->path = path;
  return static_cast<bool>(in);
}

bool WritePgm(const fs::path& path, int w, int h,
              const std::vector<unsigned char>& pixels) {
  std::ofstream out(path, std::ios::binary);
  if (!out) return false;
  out << "P5\n" << w << " " << h << "\n255\n";
  out.write(reinterpret_cast<const char*>(pixels.data()), pixels.size());
  return static_cast<bool>(out);
}

__global__ void RgbToGray(const unsigned char* rgb,
                          unsigned char* gray, int count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    int r = rgb[3 * i], g = rgb[3 * i + 1], b = rgb[3 * i + 2];
    gray[i] = static_cast<unsigned char>((77 * r + 150 * g + 29 * b) >> 8);
  }
}

int main(int argc, char** argv) {
  fs::path input_dir = argc > 1 ? argv[1] : "input";
  fs::path output_dir = argc > 2 ? argv[2] : "output";
  fs::create_directories(output_dir);

  std::vector<fs::path> files;
  for (const auto& entry : fs::directory_iterator(input_dir))
    if (entry.is_regular_file() && entry.path().extension() == ".ppm")
      files.push_back(entry.path());
  std::sort(files.begin(), files.end());
  if (files.empty()) {
    std::cerr << "No PPM images found. Generate the dataset first.\n";
    return 1;
  }

  double cpu_total_ms = 0.0, gpu_total_ms = 0.0;
  int processed = 0;

  for (const auto& file : files) {
    Image image;
    if (!ReadPpm(file, &image)) continue;
    int count = image.width * image.height;
    size_t input_bytes = image.pixels.size();
    size_t output_bytes = static_cast<size_t>(count);
    std::vector<unsigned char> gray(output_bytes);

    auto cpu_start = std::chrono::steady_clock::now();
    for (int i = 0; i < count; ++i) {
      int r = image.pixels[3*i], g = image.pixels[3*i+1], b = image.pixels[3*i+2];
      gray[i] = static_cast<unsigned char>((77*r + 150*g + 29*b) >> 8);
    }
    auto cpu_end = std::chrono::steady_clock::now();
    cpu_total_ms += std::chrono::duration<double, std::milli>(cpu_end-cpu_start).count();

    unsigned char *d_rgb = nullptr, *d_gray = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rgb), input_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gray), output_bytes));
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_rgb, image.pixels.data(), input_bytes, cudaMemcpyHostToDevice));
    int threads = 256, blocks = (count + threads - 1) / threads;
    RgbToGray<<<blocks, threads>>>(d_rgb, d_gray, count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(gray.data(), d_gray, output_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    gpu_total_ms += elapsed;
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_rgb));
    CUDA_CHECK(cudaFree(d_gray));

    if (!WritePgm(output_dir / (file.stem().string() + "_gray.pgm"),
                  image.width, image.height, gray)) return 1;
    ++processed;
  }

  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  std::cout << "CUDA devices detected: " << device_count << "\n";
  std::cout << "Images processed: " << processed << "\n";
  std::cout << "CPU processing time (sum): " << cpu_total_ms << " ms\n";
  std::cout << "GPU pipeline time (sum, includes transfers): " << gpu_total_ms << " ms\n";
  std::cout << "Output directory: " << output_dir << "\n";
  return 0;
}
