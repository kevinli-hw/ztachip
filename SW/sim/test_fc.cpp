#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include "../src/soc.h"
#include "../base/zta.h"
#include "../base/util.h"
#include "../apps/nn/tf.h"

extern "C"
{
extern int main(void);
extern void irqCallback(void);
}

int test_model_2(){
   TENSOR input;
   TENSOR output;
   Graph graph;
   ZtaStatus rc;
   TfliteNn TF2;
   int8_t* result;
   uint8_t *input_buf;

   std::vector<int> input_dim={7,3,3};
   rc = input.Create(TensorDataTypeInt8,TensorFormatSplit,TensorObjTypeRGB,input_dim);

   input_buf=(uint8_t *)input.GetBuf();
   memset(input_buf,0,7*3*3);
   for(int i = 0; i < 7; i++)
   {
     for(int j = 0; j < 3; j++)
     {
       for(int k = 0; k < 3; k++)
           input_buf[k+j*3+i*3*3]=k+i+j;
     }
   }

   assert(rc==ZtaStatusOk);
   //TF2.Create("single_fc_int8.tflite",&input,1,&output);
   TF2.Create("single_conv_3x3.tflite",&input,1,&output);
   graph.Add(&TF2);
   graph.Verify();

   FLUSH_DATA_CACHE();
   graph.Prepare();
   graph.RunUntilCompletion();
   FLUSH_DATA_CACHE();

   result = (int8_t*)output.GetBuf();
   //printf("single fc result: ");
   printf("single conv result: ");
   for (int i=0; i<10; i++)
   {
      printf("%d ", result[i]);
   }
   printf("\n");

   TF2.Unload();
   return 0;
}
int test_model(){
   printf("model test start.\n");
   TENSOR input;
   TENSOR output;
   Graph graph;
   ZtaStatus rc;
   TfliteNn TF2;
   int8_t* result;
   int8_t *input_buf;

   //std::vector<int> input_dim={3,7,7};
   //std::vector<int> input_dim={8,7,7};
   std::vector<int> input_dim={3,4,4};
   rc = input.Create(TensorDataTypeInt8,TensorFormatSplit,TensorObjTypeRGB,input_dim);
   //rc = input.Create(TensorDataTypeInt8,TensorFormatInterleaved,TensorObjTypeRGB,input_dim); //DW_conv

   //test for conv/fc
   //input_buf=(int8_t *)input.GetBuf();
   //memset(input_buf,0,3*7*7);
   //for(int i = 0; i < 3; i++)
   //{
   //  for(int j = 0; j < 7; j++)
   //  {
   //    for(int k = 0; k < 7; k++)
   //        input_buf[k+j*7+i*7*7]=k+i+j;
   //  }
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
   //test for dw_conv
   //split
   input_buf=(int8_t *)input.GetBuf();
   memset(input_buf,0,3*4*4);
   for(int i = 0; i < 3; i++)
   {
     for(int j = 0; j < 4; j++)
     {
       for(int k = 0; k < 4; k++)
           input_buf[k+j*4+i*4*4]=k+i+j;
     }
   }
   //interleaved
   //input_buf=(int8_t *)input.GetBuf();
   //memset(input_buf,0,3*4*4);

   //for(int j = 0; j < 4; j++)      // row (H)
   //{
   //   for(int k = 0; k < 4; k++)   // col (W)
   //   {
   //      for(int i = 0; i < 3; i++) // channel (C)
   //         input_buf[i + k*3 + j*4*3] = k+i+j;
   //   }
   //}


   assert(rc==ZtaStatusOk);
   //TF2.Create("single_fc_int8.tflite",&input,1,&output);
   //TF2.Create("single_conv_int8.tflite",&input,1,&output);
   //TF2.Create("mean_model.tflite",&input,1,&output);
   //TF2.Create("dw_conv.tflite",&input,1,&output);
   //TF2.Create("conv_pad_dw_int8.tflite",&input,1,&output);
   TF2.Create("one_conv.tflite",&input,1,&output);
   graph.Add(&TF2);
   graph.Verify();

   FLUSH_DATA_CACHE();
   graph.Prepare();
   graph.RunUntilCompletion();
   FLUSH_DATA_CACHE();
   {
   result = (int8_t*)output.GetBuf();
   //printf("single fc result: ");
   printf("single conv result: ");
   //printf("mean result: ");
   //for (int i=0; i<10; i++)
   for (int i=0; i<4*4*4; i++)
   //for (int i=0; i<8; i++)
   {
      printf("%d ", result[i]);
   }
   printf("\n");
   }

   TF2.Unload();
   return 0;
}
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


   graph.RunUntilCompletion();
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



int main()
{
   APB[APB_LED]=0x00000000;
   ztaInit();

   //test_model_2();
   //test_model();
   test_mobinet();

   return 0;
}

void irqCallback() {
}

