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

   std::vector<int> input_dim={3,7,7};
   rc = input.Create(TensorDataTypeInt8,TensorFormatSplit,TensorObjTypeRGB,input_dim);

   input_buf=(uint8_t *)input.GetBuf();
   memset(input_buf,0,3*7*7);
   for(int i = 0; i < 3; i++)
   {
     for(int j = 0; j < 7; j++)
     {
       for(int k = 0; k < 7; k++)
           input_buf[k+j*7+i*7*7]=k+i+j;
     }
   }

   assert(rc==ZtaStatusOk);
   TF2.Create("single_fc_int8.tflite",&input,1,&output);
   //TF2.Create("single_conv_int8.tflite",&input,1,&output);
   graph.Add(&TF2);
   graph.Verify();

   FLUSH_DATA_CACHE();
   graph.Prepare();
   graph.RunUntilCompletion();
   FLUSH_DATA_CACHE();

   TF2.Unload();
   return 0;
}

int main()
{
   APB[APB_LED]=0x00000000;
   ztaInit();

   test_model_2();
   return 0;
}

void irqCallback() {
}

