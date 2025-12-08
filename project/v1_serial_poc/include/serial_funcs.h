#ifndef __SERIAL_FUNCS__
#define __SERIAL_FUNCS__

#include "layer_structs.h"

/**
 *
 */
void conv_2d_serial_forward(tensor_t *in, tensor_t *out, conv_t *kernel);

void conv_2d_serial_backward(tensor_t *out, tensor_t *grad_out, tensor_t *grad_w);

void silu_forward(tensor_t *in, tensor_t *out);

void silu_backward(tensor_t *out, tensor_t *grad_out);

void batch_norm_forward(tensor_t *in, tensor_t *out);

void batch_norm_backward(tensor_t *out, tensor_t *grad_out);

void upsample_forward(tensor_t *in, tensor_t *out);

void upsample_backward(tensor_t *in, tensor_t *out);

void concat_forward(tensor_t *a, tensor_t *b, tensor_t *c);

/**
 * Since c2f uses multiple kernels, the assumption made is that kernels are arranged starting at 
 * the first kernel needed for the sequence and ending at the last one. The additional assumption
 * is that each conv_t struct is immediately preceeded by its data. SO the assumed data layout is
 * conv_t1 -> float[] kernel1 -> conv_t2 -> float[] kernel2 -> ...
 */
void c2f_forward(tensor_t *in, tensor_t *out, conv_t *kernel_ptr, short shortcut);

void c2f_backward(tensor_t *out_ptr, tensor_t *grad_out_ptr, tensor_t *grad_w_ptr);


#endif
