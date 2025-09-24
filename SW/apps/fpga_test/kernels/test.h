//----------------------------------------------------------------------------
// Author: Yike Li
// Date: 12/09/2025
// Description: This file is used for testing LLM extension on ztachip platform
//------------------------------------------------------------------------------


#ifndef _FPGA_TEST_H_
#define _FPGA_TEST_H_
#ifdef __cplusplus
extern "C" {
#endif

extern void kernel_fpga_test_exe(
   unsigned int _input,
   unsigned int req_id,
   unsigned int _output,
   int _src_w,
   int _src_h,
   int _src_c
);
#ifdef __cplusplus
}
#endif
#endif
