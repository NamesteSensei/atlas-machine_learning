# Convolutions and Pooling

This project focuses on implementing convolutional and pooling operations 
for image processing, particularly on grayscale and multi-channel images. 
The tasks include creating custom functions for different types of 
convolutions and pooling techniques while adhering to specific constraints.

---

## Project Structure
## Tasks Overview

### 0. Valid Convolution
- **File:** `0-convolve_grayscale_valid.py`
- **Function:** `convolve_grayscale_valid(images, kernel)`
- **Goal:** Performs a **valid** convolution on multiple grayscale images 
  using only two nested loops.

---

### 1. Same Convolution
- **File:** `1-convolve_grayscale_same.py`
- **Function:** `convolve_grayscale_same(images, kernel)`
- **Goal:** Performs a **same** convolution with padding to maintain the 
  original image size.

---

### 2. Convolution with Padding
- **File:** `2-convolve_grayscale_padding.py`
- **Function:** `convolve_grayscale_padding(images, kernel, padding)`
- **Goal:** Allows custom padding using a `(ph, pw)` tuple.

---

### 3. Strided Convolution
- **File:** `3-convolve_grayscale.py`
- **Function:** `convolve_grayscale(images, kernel, padding, stride)`
- **Goal:** Adds **stride** to the convolution, allowing more control 
  over output dimensions.

---

### 4. Convolution with Channels
- **File:** `4-convolve_channels.py`
- **Function:** `convolve_channels(images, kernel, padding, stride)`
- **Goal:** Handles convolutions on images with multiple channels (e.g., RGB).

---

### 5. Multiple Kernels
- **File:** `5-convolve.py`
- **Function:** `convolve(images, kernels, padding, stride)`
- **Goal:** Supports convolution with **multiple kernels**, producing 
  a multi-channel output.

---

### 6. Pooling
- **File:** `6-pool.py`
- **Function:** `pool(images, kernel_shape, stride, mode='max')`
- **Goal:** Implements **max** or **average pooling** on multi-channel images.

---

## How to Run the Project

1. Ensure all required files are present:
```bash
ls
0-convolve_grayscale_valid.py 0-main.py MNIST.npz README.md
