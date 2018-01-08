##Instructions for running the program - 
1. The program is hardcoded to encode the 'Lena_gray.png' image and decode it from the 'Lines.png' image. Therefore, please ensure that these files are in the same folder as the program.
2. Run 'encoding.py' to generate numpy dump files - scales.npy, offsets.npy and indices.npy. These represent the compressed version of the file
3. Run 'decoding.py' to decompress the image. This file reads the scales.npy, offsets.npy and indices.npy to recreate the image. The decompressed image is stored in the same folder as 'decompressed.png'.
4. By default, the decompression runs for 10 iterations. This can be changed by editing the iterations parameter in line:126 of the 'decoding.py' file. Increasing the number of iterations produces better quality images, decreasing it makes the decompression faster.
