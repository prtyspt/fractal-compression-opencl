import numpy as np
import matplotlib.pyplot as plt
import pyopencl as cl
import pyopencl.array
from scipy import ndimage
from scipy import misc
import time
import math
import matplotlib.pyplot as plt

###############
#OPENCL KERNEL#
###############
kernel = """

__kernel void makeCodeBlocks(__global unsigned char* input,
                             __global unsigned char* output,
                                      unsigned int size){
                             
    int i_g = get_global_id(0);
    int j_g = get_global_id(1);
     
    int i_l = get_local_id(0);
    int j_l = get_local_id(1);
     
    int i_gr = get_group_id(0);
    int j_gr = get_group_id(1);
     
    int top_corner_index = (size*2*i_g) + 2*j_g;
     
    int e00 = top_corner_index;
    int e01 = e00 + 1;
    int e10 = e00 + size;
    int e11 = e01 + size;
     
    int sum = input[e00] + input[e01] + input[e10] + input[e11];
    int avg = (unsigned char)(sum/4);
     
    int out_index = i_g*(size/2) + j_g;
     
    output[out_index] = avg;
                             
}

__kernel void decompress_update(__global unsigned char* output,
                                __global float* scale_arr,
                                __global int* offset_arr,
                                __global unsigned int* index_arr,
                                __global unsigned char* codebook,
                                unsigned int input_size,
                                unsigned int codebook_size){
                             
                             unsigned int i_g = get_global_id(0);
                             unsigned int j_g = get_global_id(1);
                             
                             unsigned int i_l = get_local_id(0);
                             unsigned int j_l = get_local_id(1);
                             
                             unsigned int i_gr = get_group_id(0);
                             unsigned int j_gr = get_group_id(1);
                             
                int input_pointer_global = (i_g*input_size + j_g)*4;
                int indices_pointer1=i_g*(input_size/2) + 2*j_g;
                int indices_pointer2=i_g*(input_size/2) + 2*j_g+1;
                unsigned int codebook_pointer_global=(index_arr[indices_pointer1]*(unsigned int)codebook_size*4)+(index_arr[indices_pointer2]*4);
                unsigned int scale_pointer=i_g*(input_size/4) + j_g;
                unsigned int offset_pointer=i_g*(input_size/4) + j_g;
                unsigned int input_pointer = input_pointer_global;
                unsigned int codebook_pointer = codebook_pointer_global;
                             
for(int i=0;i<4;i++)
{
   for(int j=0;j<4;j++) 
   {   
                 
         unsigned char codebook_pixel = codebook[codebook_pointer];
         int codebook_pixel_int = (int)codebook_pixel;
         int value = (int)(((float)codebook_pixel_int)*scale_arr[scale_pointer])+offset_arr[offset_pointer];
         if(value > 255){value = 255;}
         if(value < 0){value = 0;}
         unsigned char value_char = (char)value;
         output[input_pointer]=value;
       
         input_pointer++;
         codebook_pointer++;
         
    }            
         
    input_pointer += (input_size-4);
    codebook_pointer += (codebook_size-4);
}
                             
}


"""

####################################
#CREATING CONTEXT AND COMMAND QUEUE#
####################################
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
mf = cl.mem_flags

##################
#BUILDING PROGRAM#
##################
prg = cl.Program(ctx, kernel).build()

#################################################
#LOADING THE COMPRESSED REPRESENTATION FROM DISK#
#################################################
final_scales = np.load('scales.npy')
final_offsets = np.load('offsets.npy')
final_indices = np.load('indices.npy')

##################################
#INITIALIZING FROM A RANDOM IMAGE#
##################################
experimental2=misc.imread('lines.png')
input_size = experimental2.shape[0]

#################################################
#NO. OF ITERATIONS - INCREASE FOR BETTER QUALITY#
#################################################
iterations = 10

for k in range(iterations):
	#CREATE CODEBOOK FROM IMAGE#
    output_size = input_size/2
    out_cl = np.zeros((output_size, output_size)).astype(np.uint8)
    input_size = np.int32(input_size)
    inp_buf = cl.array.to_device(queue, experimental2)
    out_buf = cl.array.to_device(queue, out_cl)
    prg.makeCodeBlocks(queue, out_cl.shape, None, inp_buf.data, out_buf.data, input_size)
    cl.enqueue_copy(queue, out_cl, out_buf.data)
    #USE CODEBOOK AND COMPRESSED PARAMETERS TO DECODE#
    codebook_size=np.uint32(out_cl.shape[0])
    input_size=np.uint32(experimental2.shape[0])
    img_buf = cl.array.to_device(queue, experimental2)
    codebook_buf = cl.array.to_device(queue, out_cl)
    scale_buf = cl.array.to_device(queue, final_scales)
    offset_buf = cl.array.to_device(queue, final_offsets)           
    index_buf = cl.array.to_device(queue, final_indices)
    evt=prg.decompress_update(queue, final_scales.shape, None, img_buf.data, scale_buf.data, offset_buf.data, index_buf.data, codebook_buf.data, input_size, codebook_size)
    evt.wait()
    #UPDATE IMAGE#
    cl.enqueue_copy(queue, experimental2, img_buf.data)
    experimental2 = experimental2.astype(np.uint8)
    
#SAVE DECOMPRESSED IMAGE TO DISK#
plt.imshow(experimental2,cmap=plt.cm.gray)
plt.savefig('decompressed.png')