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

vint16 test::_A;
vint16 test::_Z;
vint16 test::_B;
vint32 test::_T;

_kernel_ void test::add() { 
   _VMASK=-1;
   _Z=_A+1;
}
_kernel_ void test::quant_mul() { 
   _VMASK=-1;
   _T=_A;
   //_Z=_T*_B;
   _Z=_A*_B;
}
_kernel_ void test::shift_r_v() { 
   _VMASK=-1;
   _T=_A;
   _Z=_T>>_B;
}
