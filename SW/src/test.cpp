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

#ifdef ZTACHIP_UNIT_TEST

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include "soc.h"
#include "../base/zta.h"
#include "../base/util.h"
#include "../apps/color/color.h"
#include "../apps/color/kernels/color.h"
#include "../apps/of/of.h"
#include "../apps/of/kernels/of.h"
#include "../apps/canny/canny.h"
#include "../apps/canny/kernels/canny.h"
#include "../apps/harris/harris.h"
#include "../apps/harris/kernels/harris.h"
#include "../apps/resize/resize.h"
#include "../apps/resize/kernels/resize.h"
#include "../apps/gaussian/gaussian.h"
#include "../apps/equalize/equalize.h"
#include "../apps/nn/tf.h"

// This is the test suite for ztachip
// Various vision and AI functions are tested and verified against test vectors

#define MAX_PICT_DIM 1024

#define ARRAY_ELE(p,dx,dy,x,y,elesize,offset)  ((p)[(y)*(dx)*(elesize)+(x)*(elesize)+(offset)])

// Show progress of the test using LED

static void led() {
   static int count=0;
   count++;
   LedSetState(1<<(count&0x3));
}

// Convert BGR to MONO color

static uint8_t bgr2mono(uint8_t b,uint8_t g,uint8_t r)
{
   int scale=9;
   int32_t _mono;
   int round;

   round=(1<<(scale-1));
   _mono= ((int32_t)r)*154+((int32_t)g)*302+((int32_t)b)*56;
   _mono=((_mono+round)>>scale);
   if(_mono < 0)
      _mono=0;
   else if(_mono > 255)
      _mono=255;
   return (uint8_t)_mono;
}

// Test color conversion

int test_color()
{
   ZTA_SHARED_MEM input_share;
   ZTA_SHARED_MEM output_share;
   uint8_t *input,*output;
   int src_w,src_h;
   int ii;
   uint8_t *split_result,*interleave_result;
   int srcfmt,dstfmt;
   int srcorder,dstorder;
   int dst_w,dst_h;
   int dst_x,dst_y;
   int x_off,y_off;
   int clip_w,clip_h;
   int r,c;
   int crop;
   uint8_t blue,green,red;
   TENSOR inputTensor,outputTensor;
   TensorObjType srcColorSpace,dstColorSpace;
   TensorFormat destFormat;
   Graph graph;

   input_share=ztaAllocSharedMem(4*MAX_PICT_DIM*MAX_PICT_DIM);
   output_share=ztaAllocSharedMem(4*MAX_PICT_DIM*MAX_PICT_DIM);
   input=(uint8_t *)ZTA_SHARED_MEM_VIRTUAL(input_share);
   output=(uint8_t *)ZTA_SHARED_MEM_VIRTUAL(output_share);
   split_result=(uint8_t *)malloc(MAX_PICT_DIM*MAX_PICT_DIM*3);
   interleave_result=(uint8_t *)malloc(MAX_PICT_DIM*MAX_PICT_DIM*3);

   for(src_w=16,src_h=16;src_w < 130;src_w+=2,src_h+=2) {
      for(crop=0;crop <= 8;crop++) {
      clip_w = src_w-crop;
      clip_h = src_h-crop;
      x_off=(src_w-clip_w)/2;
      y_off=(src_h-clip_h)/2;
      dst_x=0;
      dst_y=0;
      dst_w=clip_w;
      dst_h=clip_h;

      for(ii=0;ii < 16;ii++) {
    	  switch(ii) {
            case 0:
               srcfmt=kChannelFmtSplit;
               dstfmt=kChannelFmtInterleave;
               srcorder=kChannelColorBGR;
               dstorder=kChannelColorBGR;
               srcColorSpace=TensorObjTypeBGR;
               dstColorSpace=TensorObjTypeBGR;
               destFormat=TensorFormatInterleaved;
               break;
            case 1:
               srcfmt=kChannelFmtInterleave;
               dstfmt=kChannelFmtSplit;
               srcorder=kChannelColorBGR;
               dstorder=kChannelColorBGR;
               srcColorSpace=TensorObjTypeBGR;
               dstColorSpace=TensorObjTypeBGR;
               destFormat=TensorFormatSplit;
               break;
            case 2:
               srcfmt=kChannelFmtInterleave;
               dstfmt=kChannelFmtInterleave;
               srcorder=kChannelColorBGR;
               dstorder=kChannelColorBGR;
               srcColorSpace=TensorObjTypeBGR;
               dstColorSpace=TensorObjTypeBGR;
               destFormat=TensorFormatInterleaved;
               break;
            case 3:
               srcfmt=kChannelFmtSplit;
               dstfmt=kChannelFmtSplit;
               srcorder=kChannelColorBGR;
               dstorder=kChannelColorBGR;
               srcColorSpace=TensorObjTypeBGR;
               dstColorSpace=TensorObjTypeBGR;
               destFormat=TensorFormatSplit;
               break;
            case 4:
               srcfmt=kChannelFmtSplit;
               dstfmt=kChannelFmtSingle;
               srcorder=kChannelColorBGR;
               dstorder=kChannelColorMono;
               srcColorSpace=TensorObjTypeBGR;
               dstColorSpace=TensorObjTypeMonochromeSingleChannel;
               destFormat=TensorFormatSplit;
               break;
            case 5:
               srcfmt=kChannelFmtInterleave;
               dstfmt=kChannelFmtSingle;
               srcorder=kChannelColorBGR;
               dstorder=kChannelColorMono;
               srcColorSpace=TensorObjTypeBGR;
               dstColorSpace=TensorObjTypeMonochromeSingleChannel;
               destFormat=TensorFormatSplit;
               break;
            case 6:
               srcfmt=kChannelFmtSplit;
               dstfmt=kChannelFmtSingle;
               srcorder=kChannelColorRGB;
               dstorder=kChannelColorMono;
               srcColorSpace=TensorObjTypeRGB;
               dstColorSpace=TensorObjTypeMonochromeSingleChannel;
               destFormat=TensorFormatSplit;
               break;
            case 7:
               srcfmt=kChannelFmtInterleave;
               dstfmt=kChannelFmtSingle;
               srcorder=kChannelColorRGB;
               dstorder=kChannelColorMono;
               srcColorSpace=TensorObjTypeRGB;
               dstColorSpace=TensorObjTypeMonochromeSingleChannel;
               destFormat=TensorFormatSplit;
               break;
            case 8:
               srcfmt=kChannelFmtInterleave;
               dstfmt=kChannelFmtInterleave;
               srcorder=kChannelColorRGB;
               dstorder=kChannelColorMono;
               srcColorSpace=TensorObjTypeRGB;
               dstColorSpace=TensorObjTypeMonochrome;
               destFormat=TensorFormatInterleaved;
               break;
            case 9:
               srcfmt=kChannelFmtSplit;
               dstfmt=kChannelFmtSplit;
               srcorder=kChannelColorRGB;
               dstorder=kChannelColorMono;
               srcColorSpace=TensorObjTypeRGB;
               dstColorSpace=TensorObjTypeMonochrome;
               destFormat=TensorFormatSplit;
               break;
            case 10:
               srcfmt=kChannelFmtInterleave;
               dstfmt=kChannelFmtSplit;
               srcorder=kChannelColorRGB;
               dstorder=kChannelColorMono;
               srcColorSpace=TensorObjTypeRGB;
               dstColorSpace=TensorObjTypeMonochrome;
               destFormat=TensorFormatSplit;
               break;
            case 11:
               srcfmt=kChannelFmtSplit;
               dstfmt=kChannelFmtInterleave;
               srcorder=kChannelColorRGB;
               dstorder=kChannelColorMono;
               srcColorSpace=TensorObjTypeRGB;
               dstColorSpace=TensorObjTypeMonochrome;
               destFormat=TensorFormatInterleaved;
               break;
            case 12:
               srcfmt=kChannelFmtInterleave;
               dstfmt=kChannelFmtInterleave;
               srcorder=kChannelColorBGR;
               dstorder=kChannelColorMono;
               srcColorSpace=TensorObjTypeBGR;
               dstColorSpace=TensorObjTypeMonochrome;
               destFormat=TensorFormatInterleaved;
               break;
            case 13:
               srcfmt=kChannelFmtSplit;
               dstfmt=kChannelFmtSplit;
               srcorder=kChannelColorBGR;
               dstorder=kChannelColorMono;
               srcColorSpace=TensorObjTypeBGR;
               dstColorSpace=TensorObjTypeMonochrome;
               destFormat=TensorFormatSplit;
               break;
            case 14:
               srcfmt=kChannelFmtInterleave;
               dstfmt=kChannelFmtSplit;
               srcorder=kChannelColorBGR;
               dstorder=kChannelColorMono;
               srcColorSpace=TensorObjTypeBGR;
               dstColorSpace=TensorObjTypeMonochrome;
               destFormat=TensorFormatSplit;
               break;
            default:
               srcfmt=kChannelFmtSplit;
               dstfmt=kChannelFmtInterleave;
               srcorder=kChannelColorBGR;
               dstorder=kChannelColorMono;
               srcColorSpace=TensorObjTypeBGR;
               dstColorSpace=TensorObjTypeMonochrome;
               destFormat=TensorFormatInterleaved;
               break;

         }
         for(r=0;r < src_h;r++) {
            for(c=0;c < src_w;c++) {
               split_result[(r*src_w+c)]=(uint8_t)(r*100+c);
               split_result[(r*src_w+c)+src_w*src_h]=(uint8_t)(r*100+c+1);
               split_result[(r*src_w+c)+2*src_w*src_h]=(uint8_t)(r*100+c+2);

               interleave_result[3*(r*src_w+c)+0]=(uint8_t)(r*100+c);
               interleave_result[3*(r*src_w+c)+1]=(uint8_t)(r*100+c+1);
               interleave_result[3*(r*src_w+c)+2]=(uint8_t)(r*100+c+2);
            }
         }
         if(srcfmt==kChannelFmtInterleave)
            memcpy(input,interleave_result,src_w*src_h*3);
         else
            memcpy(input,split_result,src_w*src_h*3);

         ZtaStatus rc;
         std::vector<int> dim={3,src_w,src_h};
         inputTensor.Create(TensorDataTypeUint8,
                            (srcfmt==kChannelFmtInterleave)?TensorFormatInterleaved:TensorFormatSplit,
                             srcColorSpace,
                             dim);
         if(srcfmt==kChannelFmtInterleave)
            memcpy(inputTensor.GetBuf(),interleave_result,src_w*src_h*3);
         else
             memcpy(inputTensor.GetBuf(),split_result,src_w*src_h*3);
         GraphNodeColorAndReshape node;
         rc=node.Create(&inputTensor,&outputTensor,dstColorSpace,destFormat,x_off,y_off,clip_w,clip_h,dst_x,dst_y,dst_w,dst_h);
         assert(rc==ZtaStatusOk);
         graph.Clear();
         graph.Add(&node);
         rc=graph.Verify();
         FLUSH_DATA_CACHE();
         assert(rc==ZtaStatusOk);
         graph.Prepare();
         graph.RunUntilCompletion();
         FLUSH_DATA_CACHE();

         led();

         output=(uint8_t *)outputTensor.GetBuf();

         if(dstorder!=kChannelColorMono && dstfmt==kChannelFmtSplit) {
            for(r=0;r < clip_h;r++) {
               for(c=0;c < clip_w;c++) {
                  if(output[r*clip_w+c] != (uint8_t)((r+y_off)*100+(c+x_off))) {
                     exit(0);
                  }
                  if(output[clip_w*clip_h*1+r*clip_w+c] != (uint8_t)((r+y_off)*100+(c+x_off)+1)) {
                     exit(0);
                  }
                  if(output[clip_w*clip_h*2+r*clip_w+c] != (uint8_t)((r+y_off)*100+(c+x_off)+2)) {
                     exit(0);
                  }
               }
            }
         } else if(dstorder!=kChannelColorMono && dstfmt==kChannelFmtInterleave) {
            for(r=0;r < clip_h;r++) {
               for(c=0;c < clip_w;c++) {
                  if(output[3*(r*clip_w+c)+0] != (uint8_t)((r+y_off)*100+(c+x_off))) {
                     exit(0);
                  }
                  if(output[3*(r*clip_w+c)+1] != (uint8_t)((r+y_off)*100+(c+x_off)+1)) {
                     exit(0);
                  }
                  if(output[3*(r*clip_w+c)+2] != (uint8_t)((r+y_off)*100+(c+x_off)+2)) {
                     exit(0);
                  }
               }
            }
         } else {
            // Check for monochrome result....
            for(r=0;r < clip_h;r++) {
               for(c=0;c < clip_w;c++) {
                  uint8_t mono;
                  if(srcorder==kChannelColorBGR) {
                     blue=(uint8_t)((r+y_off)*100+(c+x_off));
                     green=(uint8_t)((r+y_off)*100+(c+x_off)+1);
                     red=(uint8_t)((r+y_off)*100+(c+x_off)+2);
                  } else {
                     red=(uint8_t)((r+y_off)*100+(c+x_off));
                     green=(uint8_t)((r+y_off)*100+(c+x_off)+1);
                     blue=(uint8_t)((r+y_off)*100+(c+x_off)+2);
                  }
                  mono=bgr2mono(blue,green,red);
                  if(dstfmt==kChannelFmtInterleave) {
                     if(output[3*(r*clip_w+c)+0] != mono || output[3*(r*clip_w+c)+1] != mono || output[3*(r*clip_w+c)+2] != mono) {
                        exit(0);
                     }
                  } else if(dstfmt==kChannelFmtSplit) {
                     if(output[(r*clip_w+c)] != mono ||
                        output[(r*clip_w+c)+clip_w*clip_h] != mono ||
                        output[(r*clip_w+c)+clip_w*clip_h*2] != mono) {
                        exit(0);
                     }
                  } else {
                     if(output[(r*clip_w+c)] != bgr2mono(blue,green,red)) {
                        exit(0);
                     }
                  }
               }
            }
         }
      }
   }
   }
   free(interleave_result);
   free(split_result);
   ztaFreeSharedMem(input_share);
   ztaFreeSharedMem(output_share);
   return 0;
}

// Test optical flow

int test_of() {
   char fname[200];
   FILE *fp;
   int w,h;
   int i,j;
   int src_w,src_h;
   TENSOR input[2];
   TENSOR inputCurr;
   TENSOR x_gradient;
   TENSOR y_gradient;
   TENSOR t_gradient;
   TENSOR x_vect;
   TENSOR y_vect;
   TENSOR display;
   uint8_t *input_p[2];
   int16_t *x_gradient_p;
   int16_t *y_gradient_p;
   int16_t *t_gradient_p;
   int16_t *x_vect_p;
   int16_t *y_vect_p;
   uint8_t *display_p;
   uint8_t *buf;
   uint8_t *p;
   GraphNodeOpticalFlow graphNode;
   Graph graph;
   ZtaStatus rc;

   buf=(uint8_t *)malloc(MAX_PICT_DIM*MAX_PICT_DIM*sizeof(int16_t));
   w=640;
   h=480;
   src_w=w;
   src_h=h;

   // Load the 2 images

   std::vector<int> dim={1,src_h,src_w};
   input[0].Create(TensorDataTypeUint8,TensorFormatSplit,TensorObjTypeMonochromeSingleChannel,dim);
   input[1].Create(TensorDataTypeUint8,TensorFormatSplit,TensorObjTypeMonochromeSingleChannel,dim);
   inputCurr.Create(TensorDataTypeUint8,TensorFormatSplit,TensorObjTypeMonochromeSingleChannel,dim);
   input_p[0]=(uint8_t *)input[0].GetBuf();
   input_p[1]=(uint8_t *)input[1].GetBuf();

   for(j=0;j < 2;j++) {
      sprintf(fname,"optical_flow_%d_in",j+1);
      fp=fopen(fname,"rb");
      assert(fp);
      memset(input_p[j],0,src_w*src_h);
      p=input_p[j];
      for(i=0;i < h;i++) {
         fread(p,1,w,fp);
         p+=src_w;
      }
      fclose(fp);
   }

   rc=graphNode.Create(&inputCurr,&x_gradient,&y_gradient,&t_gradient,&x_vect,&y_vect,&display);
   assert(rc==ZtaStatusOk);
   graph.Clear();
   graph.Add(&graphNode);
   rc=graph.Verify();
   assert(rc==ZtaStatusOk);

   memcpy(inputCurr.GetBuf(),input[1].GetBuf(),inputCurr.GetBufLen());
   FLUSH_DATA_CACHE();
   graph.Prepare();
   graph.RunUntilCompletion();
   FLUSH_DATA_CACHE();

   memcpy(inputCurr.GetBuf(),input[0].GetBuf(),inputCurr.GetBufLen());
   FLUSH_DATA_CACHE();
   graph.Prepare();
   graph.RunUntilCompletion();
   FLUSH_DATA_CACHE();

   x_gradient_p=(int16_t *)x_gradient.GetBuf();
   y_gradient_p=(int16_t *)y_gradient.GetBuf();
   t_gradient_p=(int16_t *)t_gradient.GetBuf();
   x_vect_p=(int16_t *)x_vect.GetBuf();
   y_vect_p=(int16_t *)y_vect.GetBuf();
   display_p=(uint8_t *)display.GetBuf();

   // Verify X-gradient
   sprintf(fname,"optical_flow_Ix.bin");
   fp = fopen(fname, "rb");
   assert(fp);
   fread(buf, 1, w*h*sizeof(int16_t), fp);
   fclose(fp);

   if(memcmp(buf,x_gradient_p,w*h*sizeof(int16_t))!=0) {
      exit(0);
   }

   // Verify Y-gradient
   sprintf(fname,"optical_flow_Iy.bin");
   fp = fopen(fname, "rb");
   assert(fp);
   fread(buf, 1, w*h*sizeof(int16_t), fp);
   fclose(fp);

   if(memcmp(buf,y_gradient_p,w*h*sizeof(int16_t))!=0) {
      exit(0);
   }

   // Verify T-gradient
   sprintf(fname,"optical_flow_It.bin");
   fp = fopen(fname, "rb");
   assert(fp);
   fread(buf, 1, w*h*sizeof(int16_t), fp);
   fclose(fp);

   if(memcmp(buf,t_gradient_p,w*h*sizeof(int16_t))!=0) {
      exit(0);
   }

   // Verify x_vect
   sprintf(fname,"optical_flow_vx.bin");
   fp = fopen(fname, "rb");
   assert(fp);
   fread(buf, 1, w*h*sizeof(int16_t), fp);
   fclose(fp);
   if(memcmp(buf,x_vect_p,w*h*sizeof(int16_t))!=0) {
      exit(0);
   }

   // Verify y_vect
   sprintf(fname,"optical_flow_vy.bin");
   fp = fopen(fname, "rb");
   assert(fp);
   fread(buf, 1, w*h*sizeof(int16_t), fp);
   fclose(fp);
   if(memcmp(buf,y_vect_p,w*h*sizeof(int16_t))!=0) {
      exit(0);
   }

   // Verify magnitude vector
   int x_mag,y_mag;
   int v;
   uint8_t rgb[3];
   for(int r=0;r<src_h;r++) {
     for(int c=0;c < src_w;c++) {
        x_mag=x_vect_p[r*src_w+c];
        y_mag=y_vect_p[r*src_w+c];
        rgb[0]=display_p[r*src_w+c];
        rgb[1]=display_p[r*src_w+c+src_w*src_h];
        rgb[2]=display_p[r*src_w+c+2*src_w*src_h];
        if(x_mag > 0)
           v=x_mag;
        else
           v=0;
        if(v>255)
           v=255;
        if(v != rgb[0]) {
           exit(0);
        }
        if(x_mag < 0)
           v=-x_mag;
        else
           v=0;
        if(v>255)
           v=255;
        if(v != rgb[1]) {
           exit(0);
        }
        if(y_mag < 0)
           v=-y_mag;
        else
           v=y_mag;
        if(v>255)
           v=255;
        if(v != rgb[2]) {
           exit(0);
        }
     }
   }
   free(buf);
   return 0;
}

// test color space conversion

int test_yuyv_to_bgr() {
   uint8_t *output;
   int src_w,src_h;
   int ii,size;
   char fname[256];
   FILE *fp;
   uint8_t *buf;
   int fmt;
   int order;
   int dst_w,dst_h;
   int x_off,y_off;
   int r,c;
   int crop;
   ZtaStatus rc;
   TENSOR inputTensor,outputTensor;
   GraphNodeColorAndReshape graphNode;
   Graph graph;

   buf=(uint8_t *)malloc(MAX_PICT_DIM*MAX_PICT_DIM*3);

   for(src_w=100,src_h=100;src_w <= 100;src_w+=2,src_h+=2) {
      if(src_w==128) {
         src_w=640;
         src_h=480;
      }
      for(crop=0;crop <= 8;crop+=2) {
         dst_w = src_w-crop;
         dst_h = src_h-crop;
         x_off=(src_w-dst_w)/2;
         x_off=(x_off/2)*2;
         y_off=(src_h-dst_h)/2;
         y_off=(y_off/2)*2;

         for(ii=0;ii < 4;ii++) {
            switch(ii) {
               case 0:
                  fmt=kChannelFmtSplit;order=kChannelColorRGB;
                  break;
               case 1:
                  fmt=kChannelFmtSplit;order=kChannelColorBGR;
                  break;
               case 2:
                  fmt=kChannelFmtInterleave;order=kChannelColorRGB;
                  break;
               default:
                  fmt=kChannelFmtInterleave;order=kChannelColorBGR;
                  break;
            }

            // Read input image...

            std::vector<int> dim={1,src_h,src_w};
            rc=inputTensor.Create(TensorDataTypeUint16,TensorFormatSplit,TensorObjTypeYUYV,dim);
            assert(rc==ZtaStatusOk);

            sprintf(fname,"color_conversion_%d_%d_in.bin",src_w,src_h);
            fp=fopen(fname,"rb");
            assert(fp);
            size=fread(inputTensor.GetBuf(),1,src_w*src_h*2,fp);
            assert(size==(src_w*src_h*2));
            fclose(fp);

            rc=graphNode.Create(
                    &inputTensor,
                    &outputTensor,
                    (order==kChannelColorRGB)?TensorObjTypeRGB:TensorObjTypeBGR,
                    (fmt==kChannelFmtSplit)?TensorFormatSplit:TensorFormatInterleaved,
                    x_off,
                    y_off,
                    dst_w,
                    dst_h);
            assert(rc==ZtaStatusOk);

            graph.Clear();
            graph.Add(&graphNode);
            rc=graph.Verify();
            assert(rc==ZtaStatusOk);
            FLUSH_DATA_CACHE();
            graph.Prepare();
            graph.RunUntilCompletion();
            FLUSH_DATA_CACHE();

            output=(uint8_t *)outputTensor.GetBuf();

            // Check against result
            sprintf(fname, "color_conversion_%s_%s_%d_%d_out.bin",
                   (order==kChannelColorBGR)?"bgr":"rgb",
                   (fmt==kChannelFmtInterleave)?"interleave":"split",
                   src_w,src_h);
            fp = fopen(fname, "rb");
            assert(fp);
            size=fread(buf, 1, src_w*src_h*3, fp);
            assert(size==(src_w*src_h*3));
            fclose(fp);

            if (fmt == kChannelFmtSplit) {
               uint8_t *p1, *p2;
               int ch;
               for (ch = 0, p1 = buf, p2 = output; ch < 3; ch++, p1 += src_w*src_h, p2 += dst_w*dst_h) {
                  for (r = 0; r < dst_h; r++) {
                     for (c = 0; c < dst_w; c++) {
                        if (ARRAY_ELE(p1, src_w, src_h, c + x_off, r + y_off, 1,0) != ARRAY_ELE(p2, dst_w, dst_h, c, r, 1,0)) {
                           exit(0);
                        }
                     }
                  }
               }
            } else {
               for (r = 0; r < dst_h; r++) {
                  for (c = 0; c < dst_w; c++) {
                     if (ARRAY_ELE(buf, src_w, src_h, c + x_off, r + y_off, 3, 0) != ARRAY_ELE(output, dst_w, dst_h, c, r, 3, 0) ||
                         ARRAY_ELE(buf, src_w, src_h, c + x_off, r + y_off, 3, 1) != ARRAY_ELE(output, dst_w, dst_h, c, r, 3, 1) ||
                         ARRAY_ELE(buf, src_w, src_h, c + x_off, r + y_off, 3, 2) != ARRAY_ELE(output, dst_w, dst_h, c, r, 3, 2)) {
                        exit(0);
                     }
                  }
               }
            }
         }
      }
   }
   free(buf);
   return 0;
}

// Test edge detection

int test_canny()
{
   char fname[200];
   FILE *fp;
   int w,h;
   int i;
   int src_w,src_h;
   int x_off,y_off;
   uint8_t *input,*output;
   uint8_t *buf;
   uint8_t *p;
   GraphNodeCanny graphNode;
   TENSOR inputTensor,outputTensor;
   ZtaStatus rc;
   Graph graph;

   w=200;
   h=200;

   for(x_off=0,y_off=0;x_off <= 0;x_off++,y_off++) {
      src_w=w+2*x_off;
      src_h=h+2*y_off;

      // Read input image...
      sprintf(fname,"canny_%d_in.bin",w);
      fp=fopen(fname,"rb");
      assert(fp);

      std::vector<int> input_dim={1,src_w,src_h};
      inputTensor.Create(TensorDataTypeUint8,TensorFormatSplit,TensorObjTypeMonochromeSingleChannel,input_dim);
      input=(uint8_t *)inputTensor.GetBuf();

      memset(input,0,src_w*src_h);
      p=input+(src_w*y_off)+x_off;
      for(i=0;i < h;i++) {
         fread(p,1,w,fp);
         p+=src_w;
      }
      fclose(fp);

      rc=graphNode.Create(&inputTensor,&outputTensor);
      assert(rc==ZtaStatusOk);

      graph.Clear();
      graph.Add(&graphNode);
      rc=graph.Verify();
      assert(rc==ZtaStatusOk);
      FLUSH_DATA_CACHE();
      graph.Prepare();
      graph.RunUntilCompletion();
      FLUSH_DATA_CACHE();

      output=(uint8_t *)outputTensor.GetBuf();

      buf=(uint8_t *)malloc(w*h);
      sprintf(fname,"canny_%d_out.bin",w);
      fp = fopen(fname, "rb");
      assert(fp);
      fread(buf, 1, w*h, fp);
      fclose(fp);
      if(memcmp(buf,output,w*h) != 0) {
         exit(0);
      }
      free(buf);
   }
   return 0;
}

// Test harris-corner point-of-interest

int test_harris() {
   char fname[200];
   FILE *fp;
   int w,h;
   int i;
   int src_w,src_h;
   int x_off,y_off;
   uint8_t *input;
   int16_t *output;
   uint8_t *buf;
   uint8_t *p;
   TENSOR inputTensor,outputTensor;
   GraphNodeHarris graphNode;
   Graph graph;
   ZtaStatus rc;

   w=200;
   h=200;

   for(x_off=0,y_off=0;x_off <= 0;x_off++,y_off++) {
      src_w=w+2*x_off;
      src_h=h+2*y_off;

      // Read input image...
      sprintf(fname,"harris_corner_%d_in.bin",w);
      fp=fopen(fname,"rb");
      assert(fp);

      std::vector<int> dim={1,src_h,src_w};
      inputTensor.Create(TensorDataTypeUint8,TensorFormatSplit,TensorObjTypeMonochromeSingleChannel,dim);
      input=(uint8_t *)inputTensor.GetBuf();

      memset(input,0,src_w*src_h);
      p=input+(src_w*y_off)+x_off;
      for(i=0;i < h;i++) {
         fread(p,1,w,fp);
         p+=src_w;
      }
      fclose(fp);

      rc=graphNode.Create(&inputTensor,&outputTensor);
      assert(rc==ZtaStatusOk);
      graph.Clear();
      graph.Add(&graphNode);
      rc=graph.Verify();
      assert(rc==ZtaStatusOk);
      FLUSH_DATA_CACHE();
      graph.Prepare();
      graph.RunUntilCompletion();
      FLUSH_DATA_CACHE();

      output=(int16_t *)outputTensor.GetBuf();

      buf=(uint8_t *)malloc(w*h*sizeof(int16_t));

      sprintf(fname,"harris_after_suppression.bin");
      fp = fopen(fname, "rb");
      assert(fp);
      fread(buf, 1, w*h*sizeof(int16_t), fp);
      fclose(fp);

      if(memcmp(buf,output,w*h*sizeof(int16_t))!=0) {
         exit(0);
      }
      free(buf);
   }
   return 0;
}

// Test image resize

static void resize(const char *fname_in,int w,int h,const char *fname_out,int dst_w,int dst_h)
{
   GraphNodeResize graph;
   Graph g;
   TENSOR input,output;
   TENSOR outputRef;
   std::vector<int> dim={3,h,w};
   ZtaStatus rc;

   input.CreateWithBitmap(fname_in);

   rc=graph.Create(&input,&output,dst_w,dst_h);
   assert(rc==ZtaStatusOk);
   g.Clear();
   g.Add(&graph);
   rc=g.Verify();
   assert(rc==ZtaStatusOk);
   FLUSH_DATA_CACHE();
   g.Prepare();
   g.RunUntilCompletion();
   FLUSH_DATA_CACHE();
   {
   char fname[100];
   uint8_t *vector=(uint8_t *)malloc(dst_w*dst_h*3);
   memset(vector,0,dst_w*dst_h*3);
   sprintf(fname,"%s",fname_out);
   outputRef.CreateWithBitmap(fname);
   if(memcmp(outputRef.GetBuf(),output.GetBuf(),dst_w*dst_h*3) != 0)
   {
      exit(0);
   }
   free(vector);
   }
}

// Test for different image resize

int test_resize() {
   resize("resize.bmp",960,540,"resize_800_400.bmp",800,400);
   resize("resize.bmp",960,540,"resize_768_300.bmp",768,300);
   resize("resize.bmp",960,540,"resize_660_256.bmp",660,256);
   resize("resize.bmp",960,540,"resize_560_200.bmp",560,200);
   resize("resize.bmp",960,540,"resize_400_180.bmp",400,180);
   resize("resize.bmp",960,540,"resize_320_172.bmp",320,172);
   resize("resize.bmp",960,540,"resize_248_140.bmp",248,140);
   return 0;
}

// Test gaussian blurring

int test_gaussian() {
   uint8_t *buf;
   FILE *fp;
   int i;
   int ch,w,h,dst_w,dst_h;
   int src_w,src_h;
   int size;
   char fname[100];
   uint8_t *input,*output,*p;
   int x_off=0;
   int y_off=0;
   TENSOR inputTensor,outputTensor;
   GraphNodeGaussian graphNode;
   ZtaStatus rc;
   Graph graph;

   buf=(uint8_t *)malloc(MAX_PICT_DIM*MAX_PICT_DIM*3);

   for(x_off=0;x_off <= 0;x_off++) {
      y_off=x_off;
      for(w=200;w <= 200;w++) {
         h=w;
         src_w=w+2*x_off;
         src_h=h+2*y_off;
         dst_w=w;
         dst_h=h;

         std::vector<int> dim={3,src_h,src_w};
         inputTensor.Create(TensorDataTypeUint8,TensorFormatSplit,TensorObjTypeRGB,dim);
         input=(uint8_t *)inputTensor.GetBuf();

         // Read input image...
         sprintf(fname,"gaussian_%d_in.bin",w);
         fp=fopen(fname,"rb");
         assert(fp);
         memset(input,0,src_w*src_h*3);

         for(ch=0;ch<3;ch++) {
            p=input+src_w*src_h*ch+(src_w*y_off)+x_off;
            for(i=0;i < h;i++) {
               fread(p,1,w,fp);
               p+=src_w;
            }
         }
         fclose(fp);

         graphNode.Create(&inputTensor,&outputTensor);

         graph.Clear();
         graph.Add(&graphNode);
         rc=graph.Verify();
         assert(rc==ZtaStatusOk);
         FLUSH_DATA_CACHE();
         graph.Prepare();
         graph.RunUntilCompletion();
         FLUSH_DATA_CACHE();

         output=(uint8_t *)outputTensor.GetBuf();

         // Check against result
         sprintf(fname, "gaussian_%d_out.bin",w);
         fp = fopen(fname, "rb");
         assert(fp);
         size=fread(buf, 1, dst_w*dst_w*3, fp);
         assert(size==(dst_w*dst_w*3));

         for(i=0;i < (dst_w*dst_h*3);i++) {
            if(buf[i] != output[i]) {
               exit(0);
            }
         }
         fclose(fp);
      }
   }
   free(buf);
   return 0;
}

// Test histogram

#define HISTOGRAM_W        256
#define HISTOGRAM_H        256
#define HISTOGRAM_CHANNEL  3

int test_histogram()
{
   uint8_t *input;
   int src_w, src_h;
   static uint8_t pict_in[HISTOGRAM_W*HISTOGRAM_H*HISTOGRAM_CHANNEL];
   static uint8_t pict_out[HISTOGRAM_W*HISTOGRAM_H*HISTOGRAM_CHANNEL];
   FILE *fp;
   int dst_w,dst_h;
   int nchannels;
   int r,c;
   TENSOR inputTensor,outputTensor;
   ZtaStatus rc;
   Graph graph;
   GraphNodeEqualize graphNode;
   uint8_t *output2;

   nchannels=HISTOGRAM_CHANNEL;

   fp = fopen("histogram_in.bin", "rb");
   assert(fp);
   fread(pict_in, 1, HISTOGRAM_W * HISTOGRAM_H * nchannels, fp);
   fclose(fp);

   fp = fopen("histogram_out.bin", "rb");
   assert(fp);
   fread(pict_out, 1, HISTOGRAM_W * HISTOGRAM_H * nchannels, fp);
   fclose(fp);

   dst_w = HISTOGRAM_W;
   dst_h = HISTOGRAM_H;
   src_w = dst_w;
   src_h = dst_h;

   std::vector<int> inputDim={nchannels,src_h,src_w};
   inputTensor.Create(TensorDataTypeUint8,TensorFormatSplit,TensorObjTypeRGB,inputDim);
   input=(uint8_t *)inputTensor.GetBuf();
   memcpy(input,pict_in,inputTensor.GetBufLen());

   rc=graphNode.Create(&inputTensor,&outputTensor);
   assert(rc==ZtaStatusOk);
   graph.Add(&graphNode);
   rc=graph.Verify();
   assert(rc==ZtaStatusOk);
   FLUSH_DATA_CACHE();
   graph.Prepare();
   graph.RunUntilCompletion();
   FLUSH_DATA_CACHE();
   graph.Prepare();
   graph.RunUntilCompletion();
   FLUSH_DATA_CACHE();

   output2=(uint8_t *)outputTensor.GetBuf();

   for (r = 0; r < (nchannels*dst_h); r++) {
      for (c = 0; c < dst_w; c++) {
         if (output2[r*dst_w + c] != pict_out[r*dst_w + c]) {
            if (ABS(output2[r*dst_w + c] - pict_out[r*dst_w + c]) >= 10) {
               exit(0);
            }
         }
      }
   }
   return 0;
}

// Test mobinet AI model

void test_mobinet()
{
   TENSOR input;
   TENSOR output;
   Graph graph;
   ZtaStatus rc;
   TfliteNn TF2;

   rc=input.CreateWithBitmap("classifier_input.bmp");
   assert(rc==ZtaStatusOk);
   TF2.Create("mobilenet_v2_1_0_224_quant.tflite",&input,1,&output);
   graph.Add(&TF2);
   graph.Verify();

   FLUSH_DATA_CACHE();
   graph.Prepare();
   //step mode
// ---- 替换 graph.RunUntilCompletion() ----
{
   const char *opNames[] = {
      "Conv2D","DepthWise","Concat","Logistic",
      "Reshape","Pad","Detect","Add","AvgPool","FC","Mean","Unknown"
   };

   int totalLayers = (int)TF2.m_operators.size();
   printf("Total layers: %d\n", totalLayers);

   for(int step = 0; step < totalLayers; step++) {

      // 1. 提交当前层到硬件（stepMode=0 = 只跑一层就返回）
      ZtaStatus rc = TF2.Execute(0, 0);

      // 2. 等待硬件完成（轮询响应队列）
      while(!TF2.AllRequestAreCompleted(0)) {
         GraphNode::CheckResponse();
      }

      // 3. 刷 cache，让 CPU 看到硬件写回的结果
      FLUSH_DATA_CACHE();

      // 4. 打印该层输出的 min/max
      NeuralNetLayer *layer = TF2.m_operators[step];
      int opCode = layer->m_def.op;
      const char *opName = (opCode >= 0 && opCode < 12) ? opNames[opCode] : "Unknown";

      if(!layer->m_def.output.empty()) {
         int outBufId = layer->m_def.output[0];
         size_t count = 1;
         int H=1, W=1, C=1;
         if(!layer->m_def.output_shape.empty() && layer->m_def.output_shape[0]) {
            for(int d : *layer->m_def.output_shape[0]) count *= (size_t)d;
            const auto &sh = *layer->m_def.output_shape[0];
            int nd = (int)sh.size();
            if      (nd >= 4) { H=sh[1]; W=sh[2]; C=sh[3]; }
            else if (nd == 3) { H=sh[1]; W=sh[2]; C=1;     }
         }

         //int minVal = 127, maxVal = -128;
         int minVal = 255, maxVal = 0;
         const int PRINT_N = 10;

         if(TF2.BufferFlatPresent(outBufId)) {
            int8_t *buf = (int8_t *)TF2.BufferGetFlat(outBufId);

            // 计算元素总数（output_shape 第0维是 batch=1，全部相乘）
            // 求 min / max
            for(size_t i = 0; i < count; i++) {
               uint8_t v = (uint8_t)buf[i];
               if(v < minVal) minVal = v;
               if(v > maxVal) maxVal = v;
            }

            printf("[Layer %3d] %-9s out_buf=%-3d count=%-7zu min=%4d max=%4d\n",
                   step, opName, outBufId, count, minVal, maxVal);
            //printf first10
            printf("[Layer %3d] %-9s first10:[ ",step, opName);
            int n = (H*W < PRINT_N) ? H*W : PRINT_N;
            const int8_t *ch_last = buf + (C - 1) * H * W;
            for (int i = 0; i < n; i++)
              printf("%d ", (uint8_t)buf[i]);  // ch0 [0..n-1] first channel
              //printf("%d ", (int)ch_last[i]);  // ch0 [0..n-1] last channel

            printf("]\n");

         } else if(TF2.BufferInterleavePresent(outBufId)) {
            int8_t *buf = (int8_t *)TF2.BufferGetInterleave(outBufId);

            // interleaved 格式有 channel 对齐 padding，实际分配比 count 大
            // 但 min/max 与顺序无关，扫 count 个有效元素即可
            for(size_t i = 0; i < count; i++) {
               uint8_t v = (uint8_t)buf[i];
                if(v < minVal) minVal = v;
                if(v > maxVal) maxVal = v;
            }
            printf("[Layer %3d] %-9s out_buf=%-3d count=%-7zu min=%4d max=%4d [interleaved]\n",
                   step, opName, outBufId, count, minVal, maxVal);

            //printf first10
            printf("[Layer %3d] %-9s first10:[ ",step, opName);
            int n = (H*W < PRINT_N) ? H*W : PRINT_N;
            for (int i = 0; i < n; i++)
              printf("%d ", (uint8_t)buf[i*C]);  // ch0 [0..n-1] first channel
              //printf("%d ", (int)buf[i * C + (C - 1)]); //last channel
            printf("]\n");
         } else {
            printf("[Layer %3d] %-9s out_buf=%-3d [no buffer]\n",
                   step, opName, outBufId);
         }
      }

      if(rc == ZtaStatusOk) break; // 最后一层完成
   }
}
// ---- 结束替换 ----


   //graph.RunUntilCompletion();
   FLUSH_DATA_CACHE();
//   {
//   size_t size=output.GetBufLen();
//   uint8_t *p=(uint8_t *)malloc(size);
//   FILE *fp=fopen("classifier.bin","rb");
//   assert(fp);
//   if(fread(p,1,size,fp) != size) {
//      assert(0);
//   }
//   if(memcmp(p,output.GetBuf(),size) != 0) {
//      assert(0);
//   }
//   printf("correct\n");
//   int top5[5];
//   uint8_t *probability=(uint8_t *)output.GetBuf();
//   NeuralNet::GetTop5(probability,output.GetBufLen(),top5);
//   for(int i=0;i < 5;i++)
//   {
////      printf("   %d %f\n",top5[i],(float)probability[top5[i]]/255.0);
//   }
//   fclose(fp);
//   free(p);
//   }

   TF2.Unload();
}

// Test SSD-Mobinet AI model

void test_mobinet_ssd()
{
   TENSOR input;
   TENSOR output[4];
   Graph graph;
   TfliteNn TF1;

   input.CreateWithBitmap("ssd_input.bmp");
   TF1.Create("detect.tflite",&input,4,&output[0],&output[1],&output[2],&output[3]);
   TF1.LabelLoad("labelmap.txt");
   graph.Add(&TF1);
   graph.Verify();
   FLUSH_DATA_CACHE();
//for(;;)
{
   graph.Prepare();
   graph.RunUntilCompletion();
   FLUSH_DATA_CACHE();
//led();
}

   // Check result
   {
   uint8_t *p;
   FILE *fp;
   size_t size;
   uint8_t *boxes=(uint8_t *)ZTA_SHARED_MEM_VIRTUAL(TF1.BufferGetInterleave(TF1.m_operators[TF1.m_operators.size()-1]->m_def.input[0]));
   uint8_t *classes=(uint8_t *)ZTA_SHARED_MEM_VIRTUAL(TF1.BufferGetInterleave(TF1.m_operators[TF1.m_operators.size()-1]->m_def.input[1]));

   fp=fopen("detect_boxes.bin","rb");
   fseek(fp, 0L, SEEK_END);
   size = ftell(fp);
   p=(uint8_t *)malloc(size);
   fseek(fp, 0L, SEEK_SET);
   if(fread(p,1,size,fp) != size) {
      assert(0);
   }
   if(memcmp(p,boxes,size) != 0) {
      assert(0);
   }
   fclose(fp);
   free(p);

   fp=fopen("detect_classes.bin","rb");
   fseek(fp, 0L, SEEK_END);
   size = ftell(fp);
   p=(uint8_t *)malloc(size);
   fseek(fp, 0L, SEEK_SET);
   if(fread(p,1,size,fp) != size) {
      assert(0);
   }
   if(memcmp(p,classes,size) != 0) {
      assert(0);
   }
   fclose(fp);
   free(p);

#if 0
   {
   float *box_p;
   float *classes_p;
   float *probability_p;
   float *numDetect_p;
   int xmin,ymin,xmax,ymax,classIdx;
   char *className;
   int numDetect;

   box_p=(float *)output[0].GetBuf();
   classes_p=(float *)output[1].GetBuf();
   probability_p=(float *)output[2].GetBuf();
   numDetect_p=(float *)output[3].GetBuf();

   numDetect=(int)numDetect_p[0];
   for(int i=0;i < (int)numDetect_p[0];i++) {
//      printf("   xmin=%f ymin=%f xmax=%f ymax=%f score=%f class=%d \n",
//      box_p[4*i+1],box_p[4*i+0],box_p[4*i+3],box_p[4*i+2],probability_p[i],(int)classes_p[i]);
      xmin=box_p[4*i+1]*300;
      ymin=box_p[4*i+0]*300;
      xmax=box_p[4*i+3]*300;
      ymax=box_p[4*i+2]*300;
      classIdx=(int)classes_p[i];
      className=(char *)TF1.LabelGet(classIdx);
   }
   }
#endif
   }
   TF1.Unload();
}

// ztachip test suite...
int test_model(){
   printf("model test start.\n");
   TENSOR input;
   TENSOR output;
   Graph graph;
   ZtaStatus rc;
   TfliteNn TF2;
   int8_t* result;
   int8_t *input_buf;

   int input_buf_cnt = 0;
   int output_buf_cnt = 0;
   //std::vector<int> input_dim={2,3,3};
   std::vector<int> input_dim={3,4,4};
   //std::vector<int> input_dim={7,3,3};
   //std::vector<int> input_dim={3,7,7};
   //std::vector<int> input_dim={8,7,7};
   rc = input.Create(TensorDataTypeInt8,TensorFormatSplit,TensorObjTypeRGB,input_dim);
   //rc = input.Create(TensorDataTypeInt8,TensorFormatInterleaved,TensorObjTypeRGB,input_dim); //DW_conv only

   //test for conv/fc
   input_buf_cnt=input_dim[0]*input_dim[1]*input_dim[2];
   output_buf_cnt = 4*4*4;
   input_buf=(int8_t *)input.GetBuf();
   memset(input_buf,0,input_buf_cnt);
   //split
   for(int i = 0; i < input_dim[0]; i++)
   {
     for(int j = 0; j < input_dim[1]; j++)
     {
       for(int k = 0; k < input_dim[2]; k++)
           input_buf[k+j*input_dim[1]+i*input_dim[1]*input_dim[2]]=(k+i+j)%128;
     }
   }
   //interleaved
   //for(int j = 0; j < 4; j++)      // row (H)
   //{
   //   for(int k = 0; k < 4; k++)   // col (W)
   //   {
   //      for(int i = 0; i < 3; i++) // channel (C)
   //         input_buf[i + k*3 + j*4*3] = k+i+j;
   //   }
   //}
   //input_buf=(int8_t *)input.GetBuf();
   //memset(input_buf,0,8*7*7);
   //for(int i = 0; i < 8; i++)
   //{
   //  for(int j = 0; j < 7; j++)
   //  {
   //    for(int k = 0; k < 7; k++)
   //    {
   //       if ((i+j+k) % 2 == 0)
   //         input_buf[k+j*7+i*7*7] = -((i*j*k)%128);
   //       else
   //         input_buf[k+j*7+i*7*7] = ((i*j*k)%128);
   //    }
   //  }
   //}


   assert(rc==ZtaStatusOk);
   //TF2.Create("single_fc_int8.tflite",&input,1,&output);
   //TF2.Create("single_conv_3x3.tflite",&input,1,&output);
   //TF2.Create("single_conv_1x1.tflite",&input,1,&output);
   //TF2.Create("dw_conv.tflite",&input,1,&output);
   //TF2.Create("conv_pad_dw_int8.tflite",&input,1,&output);
   TF2.Create("one_conv.tflite",&input,1,&output);
   //TF2.Create("pad_dw_conv.tflite",&input,1,&output);
   //TF2.Create("add_test_model_int8.tflite",&input,1,&output);
   //TF2.Create("mean_model.tflite",&input,1,&output);
   graph.Add(&TF2);
   graph.Verify();

   FLUSH_DATA_CACHE();
   graph.Prepare();
   //step mode
// ---- 替换 graph.RunUntilCompletion() ----
{
   const char *opNames[] = {
      "Conv2D","DepthWise","Concat","Logistic",
      "Reshape","Pad","Detect","Add","AvgPool","FC","Mean","Unknown"
   };

   int totalLayers = (int)TF2.m_operators.size();
   printf("Total layers: %d\n", totalLayers);

//   for (int step = 0; step < totalLayers; step++) {
//
//       // 1. 提交 / 等待 / 刷 cache（不变）
//       ZtaStatus rc = TF2.Execute(0, 0);
//       while (!TF2.AllRequestAreCompleted(0)) {
//           GraphNode::CheckResponse();
//       }
//       FLUSH_DATA_CACHE();
//
//       // 2. 获取层信息
//       NeuralNetLayer *layer = TF2.m_operators[step];
//       int opCode = layer->m_def.op;
//       const char *opName = (opCode >= 0 && opCode < 12) ? opNames[opCode] : "Unknown";
//
//       // 3. 解析 shape
//       int B = 1, H = 1, W = 1, C = 1;
//       if (!layer->m_def.output_shape.empty() && layer->m_def.output_shape[0]) {
//           const auto &sh = *layer->m_def.output_shape[0];
//           int nd = (int)sh.size();
//           if      (nd >= 4) { B=sh[0]; H=sh[1]; W=sh[2]; C=sh[3]; }
//           else if (nd == 3) { B=sh[0]; H=sh[1]; W=sh[2]; C=1;     }
//           else if (nd == 2) { B=sh[0]; H=1;     W=1;     C=sh[1]; }
//           else if (nd == 1) { B=1;     H=1;     W=1;     C=sh[0]; }
//       }
//
//       if (layer->m_def.output.empty()) {
//           printf("[Layer %3d] %s: no output, skip\n", step, opName);
//           continue;
//       }
//       int outBufId = layer->m_def.output[0];
//
//       // -------------------------------------------------------
//       // 4. 每层单独开一个文件：layer_000_Conv2D.txt
//       // -------------------------------------------------------
//       char filename[64];
//       snprintf(filename, sizeof(filename), "layer_%03d_%s.txt", step, opName);
//
//       FILE *fout = fopen(filename, "w");
//       if (!fout) {
//           printf("[Layer %3d] ERROR: cannot open %s\n", step, filename);
//           continue;
//       }
//
//       fprintf(fout, "layer=%d  op=%s  shape=[%d,%d,%d,%d]\n\n",
//               step, opName, B, H, W, C);
//
//       // 5. 写 CHW 数据
//       if (TF2.BufferFlatPresent(outBufId)) {
//           const int8_t *buf = (const int8_t *)TF2.BufferGetFlat(outBufId);
//           fprintf(fout, "format=flat(NCHW)\n");
//           for (int c = 0; c < C; c++) {
//               fprintf(fout, "\n# ch %d:\n", c);
//               for (int h = 0; h < H; h++) {
//                   for (int w = 0; w < W; w++)
//                       fprintf(fout, "%5d", (int)buf[c*H*W + h*W + w]);
//                   fprintf(fout, "\n");
//               }
//           }
//       } else if (TF2.BufferInterleavePresent(outBufId)) {
//           const int8_t *buf = (const int8_t *)TF2.BufferGetInterleave(outBufId);
//           fprintf(fout, "format=interleave(NHWC)\n");
//           for (int c = 0; c < C; c++) {
//               fprintf(fout, "\n# ch %d:\n", c);
//               for (int h = 0; h < H; h++) {
//                   for (int w = 0; w < W; w++)
//                       fprintf(fout, "%5d", (int)buf[h*W*C + w*C + c]);
//                   fprintf(fout, "\n");
//               }
//           }
//       } else {
//           fprintf(fout, "no buffer\n");
//       }
//
//       fclose(fout);  // ← 每层写完立刻关闭
//
//       printf("[Layer %3d] %-9s → %s\n", step, opName, filename);
//   }

   for(int step = 0; step < totalLayers; step++) {

      // 1. 提交当前层到硬件（stepMode=0 = 只跑一层就返回）
      ZtaStatus rc = TF2.Execute(0, 0);

      // 2. 等待硬件完成（轮询响应队列）
      while(!TF2.AllRequestAreCompleted(0)) {
         GraphNode::CheckResponse();
      }

      // 3. 刷 cache，让 CPU 看到硬件写回的结果
      FLUSH_DATA_CACHE();

      // 4. 打印该层输出的 min/max
      NeuralNetLayer *layer = TF2.m_operators[step];
      int opCode = layer->m_def.op;
      const char *opName = (opCode >= 0 && opCode < 12) ? opNames[opCode] : "Unknown";

      if(!layer->m_def.output.empty()) {
         int outBufId = layer->m_def.output[0];

         if(TF2.BufferFlatPresent(outBufId)) {
            int8_t *buf = (int8_t *)TF2.BufferGetFlat(outBufId);

            // 计算元素总数（output_shape 第0维是 batch=1，全部相乘）
            size_t count = 1;
            if(!layer->m_def.output_shape.empty() && layer->m_def.output_shape[0]) {
               for(int d : *layer->m_def.output_shape[0]) count *= (size_t)d;
            }

            // 求 min / max
            int minVal = 127, maxVal = -128;
            for(size_t i = 0; i < count; i++) {
               int v = (int)buf[i];
               if(v < minVal) minVal = v;
               if(v > maxVal) maxVal = v;
            }

            printf("[Layer %3d] %-9s out_buf=%-3d count=%-7zu min=%4d max=%4d\n",
                   step, opName, outBufId, count, minVal, maxVal);
            //printf first8
            printf("[Layer %3d] %-9s first8:[ ",step, opName);
            for (int i=0; i<count; i++)
              printf("%d ", buf[i]);
            printf("]\n");

         } else if(TF2.BufferInterleavePresent(outBufId)) {
            int8_t *buf = (int8_t *)TF2.BufferGetInterleave(outBufId);

            // interleaved 格式有 channel 对齐 padding，实际分配比 count 大
            // 但 min/max 与顺序无关，扫 count 个有效元素即可
            size_t count = 1;
            if(!layer->m_def.output_shape.empty() && layer->m_def.output_shape[0]) {
                for(int d : *layer->m_def.output_shape[0]) count *= (size_t)d;
            }

            int minVal = 127, maxVal = -128;
            for(size_t i = 0; i < count; i++) {
                int v = (int)buf[i];
                if(v < minVal) minVal = v;
                if(v > maxVal) maxVal = v;
            }
            printf("[Layer %3d] %-9s out_buf=%-3d count=%-7zu min=%4d max=%4d [interleaved]\n",
                   step, opName, outBufId, count, minVal, maxVal);

            //printf first8
            printf("[Layer %3d] %-9s first8:[ ",step, opName);
            for (int i=0; i<count; i++)
              printf("%d ", buf[i]);
            printf("]\n");
         } else {
            printf("[Layer %3d] %-9s out_buf=%-3d [no buffer]\n",
                   step, opName, outBufId);
         }
      }

      if(rc == ZtaStatusOk) break; // 最后一层完成
   }
}
//   graph.RunUntilCompletion();
   FLUSH_DATA_CACHE();
   {
   result = (int8_t*)output.GetBuf();
   //printf("single fc result: ");
   printf("single model result: ");
   //printf("single conv result: ");
   //printf("mean result: ");
   for (int i=0; i<output_buf_cnt; i++)
   //for (int i=0; i<10; i++)
   {
      printf("%d ", result[i]);
   }
   printf("\n");
   }

   TF2.Unload();
   return 0;
}

int test_mobilenet_v2(){
   printf("mobilenet test start.\n");
   TENSOR input;
   TENSOR output;
   Graph graph;
   ZtaStatus rc;
   TfliteNn TF2;
   int8_t *input_buf;

   std::vector<int> input_dim={3,224,224};
   rc = input.Create(TensorDataTypeInt8,TensorFormatSplit,TensorObjTypeRGB,input_dim);
   //rc=input.CreateWithBitmap("classifier_input.bmp");
   assert(rc==ZtaStatusOk);

   input_buf=(int8_t *)input.GetBuf();
   memset(input_buf,0,3*224*224);
   for(int i = 0; i < 3; i++)
   {
     for(int j = 0; j < 224; j++)
     {
       for(int k = 0; k < 224; k++)
           input_buf[k+j*224+i*224*224]=(k+i+j)%128;
     }
   }

   TF2.Create("mobilenet_v2_int8_bs1.tflite",&input,1,&output);
   //TF2.Create("mobilenet_v2_1_0_224_quant.tflite",&input,1,&output);
   graph.Add(&TF2);
   graph.Verify();

   FLUSH_DATA_CACHE();
   graph.Prepare();
//   graph.RunUntilCompletion();

   //step mode
// ---- 替换 graph.RunUntilCompletion() ----
{
   const char *opNames[] = {
      "Conv2D","DepthWise","Concat","Logistic",
      "Reshape","Pad","Detect","Add","AvgPool","FC","Mean","Unknown"
   };

   int totalLayers = (int)TF2.m_operators.size();
   printf("Total layers: %d\n", totalLayers);

   for(int step = 0; step < totalLayers; step++) {


    // ── 在 layer 5 执行前，打印它的输入 ──────────────────────
    if (step == 5) {
        NeuralNetLayer *dw = TF2.m_operators[5];
        NeuralNetLayer *conv3 = TF2.m_operators[3];

        int dw_inBuf   = dw->m_def.input[0];
        int conv3_outBuf = conv3->m_def.output[0];

        // 1. 先验证 buf ID 是否一致（prune 后应该相同）
        printf("[CHECK] Layer5.input[0] buf_id=%d, Layer3.output[0] buf_id=%d  %s\n",
               dw_inBuf, conv3_outBuf,
               (dw_inBuf == conv3_outBuf) ? "MATCH" : "MISMATCH!!!");

        // 2. 打印 Layer5 input 的 ch0 前10个值
        const auto &sh = *dw->m_def.input_shape[0];
        int H = sh.size()>=4?sh[1]:1, W=sh.size()>=4?sh[2]:1, C=sh.size()>=4?sh[3]:1;
        printf("[Layer  5] DW_input  H=%d W=%d C=%d  ch0_first10: ", H, W, C);

        if (TF2.BufferInterleavePresent(dw_inBuf)) {
            const int8_t *buf = (const int8_t *)TF2.BufferGetInterleave(dw_inBuf);
            for (int i = 0; i < 10; i++)
                printf("%d ", (int)buf[i * C]);  // NHWC: step=C 取 ch0
        } else if (TF2.BufferFlatPresent(dw_inBuf)) {
            const int8_t *buf = (const int8_t *)TF2.BufferGetFlat(dw_inBuf);
            for (int i = 0; i < 10; i++)
                printf("%d ", (int)buf[i]);      // NCHW: ch0 连续
        } else {
            printf("[no buffer]");
        }
        printf("\n");

        // 3. 同时打印 Layer3 output 用同一 buf_id 的值作对照
        printf("[Layer  3] Conv_output same buf  ch0_first10: ");
        if (TF2.BufferInterleavePresent(conv3_outBuf)) {
            const int8_t *buf = (const int8_t *)TF2.BufferGetInterleave(conv3_outBuf);
            const auto &sh3 = *conv3->m_def.output_shape[0];
            int C3 = sh3.size()>=4?sh3[3]:1;
            for (int i = 0; i < 10; i++)
                printf("%d ", (int)buf[i * C3]);
        } else if (TF2.BufferFlatPresent(conv3_outBuf)) {
            const int8_t *buf = (const int8_t *)TF2.BufferGetFlat(conv3_outBuf);
            for (int i = 0; i < 10; i++)
                printf("%d ", (int)buf[i]);
        }
        printf("\n");
    }

      // 1. 提交当前层到硬件（stepMode=0 = 只跑一层就返回）
      ZtaStatus rc = TF2.Execute(0, 0);

      // 2. 等待硬件完成（轮询响应队列）
      while(!TF2.AllRequestAreCompleted(0)) {
         GraphNode::CheckResponse();
      }

      // 3. 刷 cache，让 CPU 看到硬件写回的结果
      FLUSH_DATA_CACHE();

      // 4. 打印该层输出的 min/max
      NeuralNetLayer *layer = TF2.m_operators[step];
      int opCode = layer->m_def.op;
      const char *opName = (opCode >= 0 && opCode < 12) ? opNames[opCode] : "Unknown";

      if(!layer->m_def.output.empty()) {
         int outBufId = layer->m_def.output[0];
         size_t count = 1;
         int H=1, W=1, C=1;
         if(!layer->m_def.output_shape.empty() && layer->m_def.output_shape[0]) {
            for(int d : *layer->m_def.output_shape[0]) count *= (size_t)d;
            const auto &sh = *layer->m_def.output_shape[0];
            int nd = (int)sh.size();
            if      (nd >= 4) { H=sh[1]; W=sh[2]; C=sh[3]; }
            else if (nd == 3) { H=sh[1]; W=sh[2]; C=1;     }
         }

         int minVal = 127, maxVal = -128;
         const int PRINT_N = 10;

         if(TF2.BufferFlatPresent(outBufId)) {
            int8_t *buf = (int8_t *)TF2.BufferGetFlat(outBufId);

            // 计算元素总数（output_shape 第0维是 batch=1，全部相乘）
            // 求 min / max
            for(size_t i = 0; i < count; i++) {
               int v = (int)buf[i];
               if(v < minVal) minVal = v;
               if(v > maxVal) maxVal = v;
            }

            printf("[Layer %3d] %-9s out_buf=%-3d count=%-7zu min=%4d max=%4d\n",
                   step, opName, outBufId, count, minVal, maxVal);
            //printf first10
            printf("[Layer %3d] %-9s first10:[ ",step, opName);
            int n = (H*W < PRINT_N) ? H*W : PRINT_N;
            const int8_t *ch_last = buf + (C - 1) * H * W;
            for (int i = 0; i < n; i++)
              printf("%d ", (int)buf[i]);  // ch0 [0..n-1] first channel
              //printf("%d ", (int)ch_last[i]);  // ch0 [0..n-1] last channel

            printf("]\n");

         } else if(TF2.BufferInterleavePresent(outBufId)) {
            int8_t *buf = (int8_t *)TF2.BufferGetInterleave(outBufId);

            // interleaved 格式有 channel 对齐 padding，实际分配比 count 大
            // 但 min/max 与顺序无关，扫 count 个有效元素即可
            for(size_t i = 0; i < count; i++) {
                int v = (int)buf[i];
                if(v < minVal) minVal = v;
                if(v > maxVal) maxVal = v;
            }
            printf("[Layer %3d] %-9s out_buf=%-3d count=%-7zu min=%4d max=%4d [interleaved]\n",
                   step, opName, outBufId, count, minVal, maxVal);

            //printf first10
            printf("[Layer %3d] %-9s first10:[ ",step, opName);
            int n = (H*W < PRINT_N) ? H*W : PRINT_N;
            for (int i = 0; i < n; i++)
              printf("%d ", (int)buf[i*C]);  // ch0 [0..n-1] first channel
              //printf("%d ", (int)buf[i * C + (C - 1)]); //last channel
            printf("]\n");
         } else {
            printf("[Layer %3d] %-9s out_buf=%-3d [no buffer]\n",
                   step, opName, outBufId);
         }
      }

      if(rc == ZtaStatusOk) break; // 最后一层完成
   }
}
// ---- 结束替换 ----



   FLUSH_DATA_CACHE();
   {
   //size_t size=output.GetBufLen();
   //uint8_t *p=(uint8_t *)malloc(size);
   //FILE *fp=fopen("classifier.bin","rb");
   //assert(fp);
   //if(fread(p,1,size,fp) != size) {
   //   assert(0);
   //}
   //if(memcmp(p,output.GetBuf(),size) != 0) {
   //   assert(0);
   //}
   int top5[5];
   int8_t *probability=(int8_t *)output.GetBuf();
   NeuralNet::GetTop5_INT(probability,output.GetBufLen(),top5);
   for(int i=0;i < 5;i++)
   {
      printf("label: %d probability: %d\n",top5[i], probability[top5[i]]);
   }
   //fclose(fp);
   //free(p);
   }
   TF2.Unload();
   return 0;
}


int test()
{
   printf("start\n");
//   test_mobinet_ssd();
   test_mobinet();
//   test_histogram();
//   test_gaussian();
//   test_resize();
//   test_harris();
//   test_canny();
//   test_yuyv_to_bgr();
//   test_of();
//   test_color();
//   test_model();
//   test_mobilenet_v2();
   return 0;
}

#endif

