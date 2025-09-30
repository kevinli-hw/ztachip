
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include "../src/soc.h"
#include "../base/zta.h"
#include "../base/util.h"
#include "../apps/nn/tf.h"


// This is the test suite for ztachip
// Various vision and AI functions are tested and verified against test vectors

#define MAX_PICT_DIM 1024

#define ARRAY_ELE(p,dx,dy,x,y,elesize,offset)  ((p)[(y)*(dx)*(elesize)+(x)*(elesize)+(offset)])

// // Show progress of the test using LED

// static void led() {
//    static int count=0;
//    count++;
//    LedSetState(1<<(count&0x3));
// }


// Test AI model
int test_model(){
   printf("model test start.\n");
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


   //rc=input.CreateWithBitmap("classifier_input.bmp");
   assert(rc==ZtaStatusOk);
   //TF2.Create("resnet50_int8.tflite",&input,1,&output);
   TF2.Create("conv7x7_conv_int8.tflite",&input,1,&output);
   graph.Add(&TF2);
   graph.Verify();

   //FLUSH_DATA_CACHE();
   //graph.Prepare();
   //graph.RunUntilCompletion();
   //FLUSH_DATA_CACHE();
   //{
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
   //int top5[5];
   //uint8_t *probability=(uint8_t *)output.GetBuf();
   //NeuralNet::GetTop5(probability,output.GetBufLen(),top5);
   //for(int i=0;i < 5;i++)
   //{
   //   printf("label: %d probability: %f\n",top5[i],(float)probability[top5[i]]/255.0);
   //}
   //fclose(fp);
   //free(p);
   //result = (int8_t*)output.GetBuf();
   //for (int i=0; i<5; i++)
   //{
   //   printf("%d ", result[i]);
   //}
   //printf("\n");
   //}

   //TF2.Unload();
   return 0;
}

int main()
{
   test_model();

   return 0;
}
