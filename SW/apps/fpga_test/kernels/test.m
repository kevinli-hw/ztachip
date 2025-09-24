//----------------------------------------------------------------------------
// Copyright [2014] [Ztachip Technologies Inc]
//
// Author: Vuong Nguyen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except IN compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to IN writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//------------------------------------------------------------------------------

#include <stdbool.h>
#include "../../../../SW/base/ztalib.h"
#include "../../../../SW/src/soc.h"
#include "test.p.img"

#define BUFSZ (16*8*NUM_PCORE*2)

typedef struct {
   uint32_t input;
   uint32_t output;
   int src_w;
   int src_h;
   int src_c;
} REQUEST;


// Each thread is doing half the tensor
// First thread is processing the top half of tensor
// Second thread is processing the bottom half of tensor

static void test(void *_p,int pid) {
   REQUEST *req=(REQUEST *)_p;
   uint32_t from,to;

   if(pid==0) {
      from=0;
      to=req->src_h/2-1;
   } else {
      from=req->src_h/2;
      to=req->src_h-1;
   }

   >DTYPE(INT16)PCORE(NUM_PCORE)[0:NUM_PCORE-1].THREAD[0:15].test::_A[0:7] <= DTYPE(INT16)MEM(req->input,req->src_c,(pid==0)?req->src_h/2:req->src_h,req->src_w)[:][from:to][:];

   >EXE_LOCKSTEP(test::add,NUM_PCORE);

   ztaTaskYield();

   >DTYPE(INT16)MEM(req->output,req->src_c,(pid==0)?req->src_h/2:req->src_h,req->src_w)[:][from:to][:] <= DTYPE(INT16)PCORE(NUM_PCORE)[0:NUM_PCORE-1].THREAD[0:15].test::_Z[0:7];
}

//
// This is a simple test
// It add 1 to every elements of a tensor
//

void kernel_fpga_test_exe(
	unsigned int _input,
	unsigned int req_id,
	unsigned int _output,
  int _src_w,
  int _src_h,
  int _src_c
		) {
   REQUEST req; 
   int i;

   ztaInitPcore(zta_pcore_img);

   req.input=_input;
   req.output=_output;
   req.src_w=_src_w;
   req.src_h=_src_h;
   req.src_c=_src_c;
   
   ztaDualHartExecute(test,&req);

   ztaJobDone(req_id);

}
