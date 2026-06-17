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

#include "../../../base/zta.h"
#include "fcn.h"

//---  Process fully-connected layer (innerproduct)

_share int16 inner_product::bot[IP_CHUNK_SIZE];
vint16 inner_product::coef[IP_CHUNK_SIZE];
vint16 inner_product::top;
int inner_product::out_scale;
vint32 inner_product::_A;
vint16 inner_product::biasHi;
vint16 inner_product::biasLo;
vint16 inner_product::activation_multiplier;
vint16 inner_product::activation_shift;

_kernel_ void inner_product::init(int _out_scale) {
   out_scale=_out_scale;
}

_kernel_ void inner_product::start() {
   _A = biasHi;
   _A = (_A << (DATA_BIT_WIDTH-1));
   _A += biasLo;
}

_kernel_ void inner_product::exe() {
   int i;
#pragma unroll
   for(i=0;i < IP_CHUNK_SIZE;i++) {
      _A += bot[i]*coef[i];
   }
}

// Activation

_kernel_ void inner_product::activate_none() {
   top = _A >> out_scale;
}

_kernel_ void inner_product::activate_per_channel() {
   _A = _A * activation_multiplier;
   top = _A >> activation_shift;
}


// ---- Pooling layer...

vint16 max_pool::bot[POOL_BOT_SIZE];
vint32 max_pool::_A;
int max_pool::out_scale;
vint16 max_pool::top;

_kernel_ void max_pool::init(int _out_scale) {
   out_scale=_out_scale;
   _A=0;
}

// Max pooling: initialize the per-lane running max to the pad/min value, and
// prefill the whole window buffer so unused slots (when ksz*ksz < POOL_BOT_SIZE)
// dont corrupt the result. Uses the normal vector register 'top', NOT the accumulator.
_kernel_ void max_pool::init_max(int _pad_val) {
   int i;
   top = _pad_val;
#pragma unroll
   for(i=0;i < POOL_BOT_SIZE;i++)
      bot[i] = _pad_val;
}

// Do pooling averate.
// Since pcore cannot do division, just do addition here
// Divide for averaging is done by stream processor later
 
_kernel_ void max_pool::exe() {
   vint16 *p;
   int i;
   p=bot;
#pragma unroll
   for(i=0;i < POOL_BOT_SIZE;i++) {
      _A += p[0];
      p++;
   }
}

// Seed running max from bot[0] (reliable vint16 copy), then fold bot[1..7].
// init_max must be called beforehand to fill bot[ksz..7] with pad_val.
_kernel_ void max_pool::exe_max_first() {
   int i;
   top = bot[0];
#pragma unroll
   for(i=1;i < POOL_BOT_SIZE;i++) {
      _VMASK = LT(top,bot[i]); top = bot[i]; _VMASK = -1;
   }
}

// Fold bot[0..7] into running max.
_kernel_ void max_pool::exe_max() {
   int i;
#pragma unroll
   for(i=0;i < POOL_BOT_SIZE;i++) {
      _VMASK = LT(top,bot[i]); top = bot[i]; _VMASK = -1;
   }
}

_kernel_ void max_pool::finish() {
   top = _A >> out_scale;
   _A = 0;
}


// --- Do concatenation layer.

_share int16 concatenate::buf[CONCATENATE_BUFSZ];

// No pcore code required for concatenation layer
// Concatenation layer is processed by mcore and stream processor only

_kernel_ void concatenate::dummy(int16 _dummy) {
   buf[0]=_dummy;
}
