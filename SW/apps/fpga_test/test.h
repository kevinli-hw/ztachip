//----------------------------------------------------------------------------
// Author: Yike Li
// Date: 12/09/2025
// Description: This file is used for testing LLM extension on ztachip platform
//------------------------------------------------------------------------------

#ifndef _TARGET_APPS_FPGA_TEST_H_
#define _TARGET_APPS_FPGA_TEST_H_

#include "../../base/tensor.h"
#include "../../base/graph.h"

class GraphNodeTest : public GraphNode {
public:
   GraphNodeTest();
   GraphNodeTest(TENSOR *input, TENSOR *output);
   virtual ~GraphNodeTest();
   ZtaStatus Create(TENSOR *input, TENSOR *output);
   virtual ZtaStatus Verify();
   virtual ZtaStatus Execute(int queue,int stepMode);
private:
   void Cleanup();
private:
   int m_w;
   int m_h;
   int m_nChannel;
   TENSOR *m_input;
   TENSOR *m_output;
};

#endif
