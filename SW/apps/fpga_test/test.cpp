//----------------------------------------------------------------------------
// Author: Yike Li
// Date: 12/09/2025
// Description: This file is used for testing LLM extension on ztachip platform
//------------------------------------------------------------------------------

#include <math.h>
#include "../../base/types.h"
#include "../../base/tensor.h"
#include "../../base/graph.h"
#include "kernels/test.h"
#include "test.h"

#ifndef BUFSZ
#define BUFSZ 1024

GraphNodeTest::GraphNodeTest() {
}

GraphNodeTest::GraphNodeTest(TENSOR *input, TENSOR *output) : GraphNodeTest() {
   Create(input,output);
}

GraphNodeTest::~GraphNodeTest() {
   Cleanup();
}

ZtaStatus GraphNodeTest::Create(TENSOR *input, TENSOR *output) {
   Cleanup();
   m_input=input;
   m_output=output;
   return ZtaStatusOk;
}


ZtaStatus GraphNodeTest::Verify() {
   if((*(m_input->GetDimension())).size() != 3)
      return ZtaStatusFail;
   m_w=(*(m_input->GetDimension()))[2];
   m_h=(*(m_input->GetDimension()))[1];
   m_nChannel=(*(m_input->GetDimension()))[0];
   if(m_nChannel != 1)
      return ZtaStatusFail;

   std::vector<int> dim={m_nChannel,m_h,m_w};
   m_output->Create(TensorDataTypeUint16,TensorFormatSplit,TensorObjTypeMonochromeSingleChannel,dim);
   return ZtaStatusOk;
}

ZtaStatus GraphNodeTest::Execute(int queue,int stepMode) {
   kernel_fpga_test_exe(
      (unsigned int)m_input->GetBuf(),
      (unsigned int)GetJobId(queue),
      (unsigned int)m_output->GetBuf(),
      m_w,
      m_h,
      m_nChannel
   );
   return ZtaStatusOk;
}

void GraphNodeTest::Cleanup() {
}

#endif
