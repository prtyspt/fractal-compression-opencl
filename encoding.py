import numpy as np
import pyopencl as cl
from scipy import ndimage
from scipy import misc
import matplotlib.pyplot as plt


#######
#INPUT#
#######
input_img = misc.imread('lena_gray.png')
input_size = input_img.shape[0]
input_size = np.int32(input_size)

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


__kernel void encode(__global unsigned char* input,
                     __global unsigned char* codebook,
                     __global          float* scale_arr,
                     __global          int* offset_arr,
                     __global unsigned int* index_arr,
                              unsigned int  input_size,
                              unsigned int  codebook_size){
                     
    int i_g = get_global_id(0);
    int j_g = get_global_id(1);
             
    int i_l = get_local_id(0);
    int j_l = get_local_id(1);
             
    int i_gr = get_group_id(0);
    int j_gr = get_group_id(1);

    unsigned int min = (unsigned int)(-1);
    int final_offset;
    float final_scale;
    int indexx, indexy;

    int input_pointer_global = (i_g*input_size + j_g)*4;
    //printf("%d, %d \\n", i_g, j_g);
    //if(i_g==0 && j_g==0){printf(" %u, %hhu \\n", (unsigned int)input[input_pointer_global], input[input_pointer_global]);}
    

    for(int k=0;k<codebook_size;k+=4){

        for(int l=0;l<codebook_size;l+=4){

            int codebook_pointer_global = (k*codebook_size + l);
            int input_pointer = input_pointer_global;
            int codebook_pointer = codebook_pointer_global;
            
            unsigned long int code_block_sum = 0;
            unsigned long int code_block_squared_sum = 0;
            unsigned long int image_sum = 0;
            unsigned long int cross_sum = 0;
            int i, j;
            for(i=0;i<4;i++){

                for(j=0;j<4;j++){

                    unsigned long int current_codeblock_pixel = (unsigned long int)codebook[codebook_pointer];
                    
                    unsigned long int current_image_pixel = (unsigned long int)input[input_pointer];
                    //if(i_g==0 && j_g==0){printf(" %lu, %hhu \\n", current_image_pixel, (unsigned char)current_image_pixel);}
                    code_block_sum += current_codeblock_pixel;
                    code_block_squared_sum += (current_codeblock_pixel*current_codeblock_pixel);
                    image_sum += current_image_pixel;
                    cross_sum += (current_image_pixel*current_codeblock_pixel);

                    input_pointer++;
                    codebook_pointer++;

                }

                input_pointer += (input_size-4);
                codebook_pointer += (codebook_size-4);

            }
            
            
            
            //printf("%li", cross_sum);
            //printf("%li", image_sum);
            //printf("%li", code_block_sum);
            //printf("%li", code_block_squared_sum);

            long int num_scale = (long int)(16*cross_sum - image_sum*code_block_sum);
            long int denom = (long int)(16*code_block_squared_sum - code_block_sum*code_block_sum);
            
            float scale = (float)((float)(num_scale)/(float)(denom));

            int num_off = (int)((float)image_sum - (scale*(float)code_block_sum));
            int offset = num_off/16;
            
            if(i_g==0 && j_g==0 && scale<(-1.0f)){printf("%lu,%lu,%lu,%lu, S %li,%li,%f O %d,%d\\n", cross_sum, image_sum, code_block_sum, code_block_squared_sum, num_scale, denom, scale, num_off, offset);}

            int codebook_pointer_new = codebook_pointer_global;
            int input_pointer_new = input_pointer_global;

            unsigned int distance = 0;

            for(int i=0;i<4;i++){

                for(int j=0;j<4;j++){
                    
                    int x = (scale*codebook[codebook_pointer_new]) + (float)offset - input[input_pointer_new];
                    distance += (unsigned int)(x*x);
                    codebook_pointer_new++;
                    input_pointer_new++;

                }
                
                input_pointer_new += (input_size-4);
                codebook_pointer_new += (codebook_size-4);
            
            }
            
            if(distance<(float)min){
                min=(unsigned int)distance;
                final_scale = scale;
                final_offset = offset;
                indexx = k/4;
                indexy = l/4;
            
            }

        }

    }

    offset_arr[i_g*(input_size/4) + j_g] = final_offset;
    scale_arr[i_g*(input_size/4) + j_g] = final_scale;
    index_arr[i_g*(input_size/2) + 2*j_g] = indexx;
    index_arr[i_g*(input_size/2) + 2*j_g + 1] = indexy;
                     
}
"""
###################
#INITIALIZE OUTPUT#
###################
output_size = input_size/2
out_cl = np.zeros((output_size, output_size)).astype(np.uint8)

####################################
#CREATING CONTEXT AND COMMAND QUEUE#
####################################
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
mf = cl.mem_flags

######################
#CREATING I/O BUFFERS#
######################
inp_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_img)
out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, out_cl.nbytes)

##################
#BUILDING PROGRAM#
##################
prg = cl.Program(ctx, kernel).build()

######################
#CALLING THE FUNCTION#
######################
prg.makeCodeBlocks(queue, out_cl.shape, None, inp_buf, out_buf, input_size)

#########################################
#RETRIEVING THE CODEBOOK FROM THE DEVICE#
#########################################
cl.enqueue_copy(queue, out_cl, out_buf)

##################################
#INITIALIZING OUTPUT FOR ENCODING#
##################################
final_scales = np.zeros((input_size/4, input_size/4), dtype=np.float32)
final_offsets = np.zeros((input_size/4, input_size/4), dtype=np.int32)
final_indices = np.zeros(((input_size/4, input_size/2)), dtype=np.int32)
input_img = input_img.astype(np.uint8) 
out_cl = out_cl.astype(np.uint8)

######################
#CREATING I/O BUFFERS#
######################
input_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_img)
codebook_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=out_cl)
scale_buf = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=final_scales)
offset_buf = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=final_offsets)
index_buf = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=final_indices)

codebook_size = np.int32(input_size/2)

######################
#CALLING THE FUNCTION#
######################
prg.encode(queue, final_scales.shape, None, input_buf, codebook_buf, scale_buf, offset_buf, index_buf, input_size, codebook_size)

##########################################
#RETRIEVING THE COMPRESSED REPRESENTATION#
##########################################
cl.enqueue_copy(queue, final_scales, scale_buf)
cl.enqueue_copy(queue, final_offsets, offset_buf)
cl.enqueue_copy(queue, final_indices, index_buf)

##############################################
#SAVING THE COMPRESSED REPRESENTATION TO DISK#
##############################################
np.save('indices', final_indices)
np.save('offsets', final_offsets)
np.save('scales', final_scales)




