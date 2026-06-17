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

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <vector>
#include "../../base/types.h"
#include "../../base/util.h"
#include "../../base/ztalib.h"
#include "kernels/fcn.h"
#include "nn_poolmax.h"

// Do max pool layer

NeuralNetLayerPoolMax::NeuralNetLayerPoolMax(NeuralNet *nn,NeuralNetOperatorDef* def) : NeuralNetLayer(nn,def) {
}

NeuralNetLayerPoolMax::~NeuralNetLayerPoolMax() {
}

ZtaStatus NeuralNetLayerPoolMax::Prepare() {
   m_shmSpu=ztaBuildSpuBundle(1,SpuMaxPool,this,0,0);
   m_nn->BufferAllocate(m_shmSpu);
   return ZtaStatusOk;
}

ZtaStatus NeuralNetLayerPoolMax::Evaluate(int queue) {
   NeuralNetOperatorDef *op=&m_def;
   bool interleave=(m_nn->BufferGetInterleave(op->output[0])!=0);

   kernel_Pooling_exe(
      (unsigned int)GetJobId(queue),
      (op->input_type[0] == NeuralNetTensorType_INT8) ? 1 : 0,
      2, // is_avg_pool=2 means max pool
      (unsigned int)(interleave?m_nn->BufferGetInterleave(op->input[0]):m_nn->BufferGetFlat(op->input[0])),
      (unsigned int)(interleave?m_nn->BufferGetInterleave(op->output[0]):m_nn->BufferGetFlat(op->output[0])),
      op->u.pool_avg.filter_w,
      op->u.pool_avg.stride_w,
      (*op->output_shape[0])[3],
      (*op->output_shape[0])[2],
      (*op->input_shape[0])[3],
      (*op->input_shape[0])[2],
      (unsigned int)m_shmSpu,
      0,
      0,
      op->u.pool_avg.pad_w);
   return ZtaStatusOk;
}

//int16_t NeuralNetLayerPoolMax::SpuMaxPool(int16_t _in,void *pparm,uint32_t parm,uint32_t parm2)
//{
//   NeuralNetLayer *layer=static_cast<NeuralNetLayer *>(pparm);
//   NeuralNetOperatorDef *op=layer?&((NeuralNetLayerPoolMax *)layer)->m_def:0;
//   static int activation_max=0;
//   static int activation_min=0;
//   if(op) {
//      activation_max=op->u.pool_avg.activation_max;
//      activation_min=op->u.pool_avg.activation_min;
//      ((NeuralNetLayerPoolMax *)layer)->m_outputShift=0;
//   }
//   if(_in > activation_max) {
//      return static_cast<int16_t>(activation_max);
//   } else if(_in < activation_min) {
//      return static_cast<int16_t>(activation_min);
//   } else {
//      return _in;
//   }
//}
int16_t NeuralNetLayerPoolMax::SpuMaxPool(int16_t _in,void *pparm,uint32_t parm,uint32_t parm2)
{
   NeuralNetLayer *layer=static_cast<NeuralNetLayer *>(pparm);
   if(layer)
      ((NeuralNetLayerPoolMax *)layer)->m_outputShift=0;
   // Max pooling outputs one of its input values, which is already a valid
   // quantized number in range. Do NOT clamp to [activation_min, activation_max]:
   // activation_min was 0 here, which wrongly forced valid negative inputs
   // (e.g. -2, or conv+ReLU's negative quantized outputs) up to 0. Plain max
   // pool has no activation of its own; just pass the value through.
   return _in;
}
