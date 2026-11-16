NVCC ?= nvcc
NVCCFLAGS ?= -O2 -std=c++17
TARGET = batch_grayscale
SOURCE = src/batch_grayscale.cu

.PHONY: all clean dataset run
all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

dataset:
	python3 scripts/generate_dataset.py --count 200 --width 256 --height 256

run: $(TARGET)
	./$(TARGET) data/input data/output

clean:
	rm -f $(TARGET)
	rm -f data/output/*.pgm
