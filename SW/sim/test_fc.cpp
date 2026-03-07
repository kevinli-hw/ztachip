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
   std::vector<int> input_dim={8,7,7};
   rc = input.Create(TensorDataTypeInt8,TensorFormatSplit,TensorObjTypeRGB,input_dim);

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
   input_buf=(int8_t *)input.GetBuf();
   memset(input_buf,0,8*7*7);
   for(int i = 0; i < 8; i++)
   {
     for(int j = 0; j < 7; j++)
     {
       for(int k = 0; k < 7; k++)
       {
          if ((i+j+k) % 2 == 0)
            input_buf[k+j*7+i*7*7] = -((i*j*k)%128);
          else
            input_buf[k+j*7+i*7*7] = ((i*j*k)%128);
       }
     }
   }


   assert(rc==ZtaStatusOk);
   //TF2.Create("single_fc_int8.tflite",&input,1,&output);
   //TF2.Create("single_conv_int8.tflite",&input,1,&output);
   TF2.Create("mean_model.tflite",&input,1,&output);
   graph.Add(&TF2);
   graph.Verify();

   FLUSH_DATA_CACHE();
   graph.Prepare();
   graph.RunUntilCompletion();
   FLUSH_DATA_CACHE();
   {
   result = (int8_t*)output.GetBuf();
   //printf("single fc result: ");
   //printf("single conv result: ");
   printf("mean result: ");
   //for (int i=0; i<10; i++)
   for (int i=0; i<8; i++)
   {
      printf("%d ", result[i]);
   }
   printf("\n");
   }

   TF2.Unload();
   return 0;
}


int main()
{
   APB[APB_LED]=0x00000000;
   ztaInit();

   test_model_2();
   //test_model();
   return 0;
}

void irqCallback() {
}

