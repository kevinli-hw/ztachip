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

#include <stdint.h>
#include <assert.h>
#include <vector>
#include <stdarg.h>
#include <malloc.h>
#include <math.h>
#include "types.h"
#include "ztalib.h"
#include "util.h"
#include "tensor.h"

// Default constructor.
// Tensor definition/allocation to be provided later

TENSOR::TENSOR() {
   m_shm=0;
   m_isAlias=false;
   m_dataType=TensorDataTypeUint8;
   m_dataElementLen=1;
   m_fmt=TensorFormatSplit;
   m_objType=TensorObjTypeUnknown;
   m_buf=0;
   m_size=0;
}

// Create a fully defined tensor.
// Also allocate memory for it

TENSOR::TENSOR(TensorDataType _dataType,TensorFormat _fmt,TensorObjType _objType,int numDim,...):TENSOR() {
   va_list args;
   int v;

   setDataType(_dataType);
   setFormat(_fmt);
   setObjType(_objType);
   va_start(args,numDim);
   for(int i=0;i < numDim;i++) {
      v=va_arg(args,int);
      m_dim.push_back(v);
   }
   m_size=TENSOR::GetTensorSize(m_dim)*m_dataElementLen;
   allocate();
   va_end(args);
}

// Create tensor. Free previous allocation
// If _shm specified, tensor buffer has been created already so just reference it directly
// Otherwise allocate new buffer

ZtaStatus TENSOR::Create(TensorDataType _dataType,TensorFormat _fmt,TensorObjType _objType,std::vector<int> &dim,
                        ZTA_SHARED_MEM _shm) {
   if(m_shm && !m_isAlias) {
      ztaFreeSharedMem(m_shm);
   }
   m_shm=0;
   m_isAlias=false;
   setDataType(_dataType);
   setFormat(_fmt);
   setObjType(_objType);
   setDimension(dim);
   if(_shm)
      allocate(_shm);
   else
      allocate();
   return ZtaStatusOk;
}

ZtaStatus TENSOR::Clone(TENSOR *other) {
   return Create(other->GetDataType(),other->GetFormat(),other->GetObjType(),other->m_dim);
}

int8_t TENSOR::getQuantizedInput(uint8_t data)
{
   const float scale = 0.007843137718737125f;
   const int zero_point = -1;
   float data_f = ((float)data) / 127.5f - 1.0f;
   int data_q = (int)roundf(data_f / scale) + zero_point;
   if (data_q > 127)
       data_q = 127;
   else if (data_q < -128)
       data_q = -128;
   return (int8_t)data_q;
}

ZtaStatus TENSOR::CreateWithBitmap(const char *bmpFile,TensorFormat fmt, TensorDataType dataType)
{
   uint8_t *pict;
   int bmp_w,bmp_h;
   int bmpActualWidth;
   int r,c;
   int w,h;
   int dx,dy;

   pict = bmpRead(bmpFile,&bmp_h,&bmp_w);
   if(!pict) {
      return ZtaStatusFail;
   }
   if(fmt==TensorFormatSplit) {
      std::vector<int> dim={3,bmp_h,bmp_w};
      //Create(TensorDataTypeUint8,TensorFormatSplit,TensorObjTypeRGB,dim);
      Create(dataType,TensorFormatSplit,TensorObjTypeRGB,dim);
   } else {
      std::vector<int> dim={bmp_h,3*bmp_w};
      //Create(TensorDataTypeUint8,TensorFormatSplit,TensorObjTypeRGB,dim);
      Create(dataType,TensorFormatSplit,TensorObjTypeRGB,dim);
   }

   w=bmp_w;
   h=bmp_h;
   dx=0;
   dy=0;
   bmpActualWidth=((bmp_w*3+3)/4)*4;
   if (dataType == TensorDataTypeUint8)
   {
      uint8_t *output=(uint8_t *)GetBuf();
      uint8_t red, blue, green;
      for (r = 0; r < h; r++) {
         for (c = 0; c < w; c++) {
            blue = (pict[((bmp_h-1)-(r+dy))*bmpActualWidth+3*(c+dx)+0]);
            green = (pict[((bmp_h-1)-(r+dy))*bmpActualWidth+3*(c+dx)+1]);
            red = (pict[((bmp_h-1)-(r+dy))*bmpActualWidth+3*(c+dx)+2]);
            if(fmt==TensorFormatSplit) {
               output[0*w*h+r*w+c] = red;
               output[1*w*h+r*w+c] = green;
               output[2*w*h+r*w+c] = blue;
            } else {
                output[3*(r*w+c)] = red;
                output[3*(r*w+c)+1] = green;
                output[3*(r*w+c)+2] = blue;
            }
         }
      }
   }else{
      int8_t *output=(int8_t *)GetBuf();
      uint8_t red, blue, green;
      for (r = 0; r < h; r++) {
         for (c = 0; c < w; c++) {
            blue = (pict[((bmp_h-1)-(r+dy))*bmpActualWidth+3*(c+dx)+0]);
            green = (pict[((bmp_h-1)-(r+dy))*bmpActualWidth+3*(c+dx)+1]);
            red = (pict[((bmp_h-1)-(r+dy))*bmpActualWidth+3*(c+dx)+2]);

            //tflite quantized input
            if(fmt==TensorFormatSplit) {
               output[0*w*h+r*w+c] = getQuantizedInput(red);
               output[1*w*h+r*w+c] = getQuantizedInput(green);
               output[2*w*h+r*w+c] = getQuantizedInput(blue);
            } else {
               output[3*(r*w+c)]   = getQuantizedInput(red);
               output[3*(r*w+c)+1] = getQuantizedInput(green);
               output[3*(r*w+c)+2] = getQuantizedInput(blue);
            }

            //simplified input
            //if(fmt==TensorFormatSplit) {
            //   output[0*w*h+r*w+c] = (int8_t)(red-128);
            //   output[1*w*h+r*w+c] = (int8_t)(green-128);
            //   output[2*w*h+r*w+c] = (int8_t)(blue-128);
            //} else {
            //    output[3*(r*w+c)] = (int8_t)(red-128);
            //    output[3*(r*w+c)+1] = (int8_t)(green-128);
            //    output[3*(r*w+c)+2] = (int8_t)(blue-128);
            //}
         }
      }
   }
   free(pict);
   return ZtaStatusOk;
}

TENSOR::~TENSOR() {
   if(m_shm && !m_isAlias)
      ztaFreeSharedMem(m_shm);
}

ZtaStatus TENSOR::setDataType(TensorDataType _dataType) {
   m_dataType=_dataType;
   switch(m_dataType) {
      case TensorDataTypeInt8:
      case TensorDataTypeUint8:
         m_dataElementLen=1;
         break;
      case TensorDataTypeInt16:
      case TensorDataTypeUint16:
         m_dataElementLen=2;
         break;
      case TensorDataTypeFloat32:
         m_dataElementLen=4;
         break;
      default:
         assert(0);
   }
   return ZtaStatusOk;
}

ZtaStatus TENSOR::setObjType(TensorObjType _objType) {
   m_objType=_objType;
   return ZtaStatusOk;
}

ZtaStatus TENSOR::setFormat(TensorFormat fmt) {
   m_fmt=fmt;
   return ZtaStatusOk;
}

ZtaStatus TENSOR::setDimension(std::vector<int> &_dim) {
   m_dim.clear();
   m_dim=_dim;
   m_size = TENSOR::GetTensorSize(m_dim)*m_dataElementLen;
   return ZtaStatusOk;
}

ZtaStatus TENSOR::allocate(ZTA_SHARED_MEM shm) {
   assert(m_size < (1<<(DP_ADDR_WIDTH-1))); // Tensor must be less than half tensor address dynamic range
   if(m_shm && !m_isAlias)
      ztaFreeSharedMem(m_shm);
   m_isAlias=false;
   if(shm) {
      m_shm=shm;
   } else {
      m_shm=ztaAllocSharedMem(m_size);
   }
   m_buf=ZTA_SHARED_MEM_VIRTUAL(m_shm);
   return ZtaStatusOk;
}

// Set this tensor as an alias for another buffer

ZtaStatus TENSOR::Alias(TENSOR *other) {
   if(m_shm && !m_isAlias)
      ztaFreeSharedMem(m_shm);
   m_shm=other->GetShm();
   m_isAlias=true;
   m_buf=ZTA_SHARED_MEM_VIRTUAL(m_shm);
   assert(other->m_size==m_size);
   return ZtaStatusOk;
}

ZtaStatus TENSOR::Alias(ZTA_SHARED_MEM _shm) {
   if(m_shm && !m_isAlias)
      ztaFreeSharedMem(m_shm);
   m_shm=_shm;
   m_isAlias=true;
   m_buf=ZTA_SHARED_MEM_VIRTUAL(m_shm);
   return ZtaStatusOk;
}

// Return total tensor array size

size_t TENSOR::GetTensorSize(std::vector<int>& shape) {
   size_t sz=1;
   for(int i=0;i < (int)shape.size();i++) {
      sz*=shape[i];
   }
   return sz;
}

