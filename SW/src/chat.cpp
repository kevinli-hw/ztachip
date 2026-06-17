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
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include "soc.h"
#include "../base/zta.h"
#include "../base/util.h"
#include "../apps/llm/llm.h"


#ifndef __WIN32__
static char *getInput()
{
    static int inputLen=0;
    static char input[256];
    char ch;

    printf("\r\n>");
    fflush(stdout);
    while(UartReadAvailable())
        ch = UartRead();
    inputLen = 0;
    for(;;) {    
        if(UartReadAvailable()) {
            ch = UartRead();
            printf("%c",ch);
            fflush(stdout);
            if(ch==0x3) {
                input[0] = 0x3;
                input[1] = 0;
                return input;
            }
            else if(ch=='\n' || ch=='\r') {
                printf("\r\n");
                fflush(stdout);
                input[inputLen]=0;
                return input;
            } else if(ch=='\b') {
                if(inputLen > 0)
                    inputLen--;
            } else {
                if(inputLen < (int)(sizeof(input)-1)) {
                    input[inputLen++]=ch;
                }
            } 
        }
    }
    return 0;
}
#endif

static llama ai;


int chat() {
    static std::string output_ref_0,output_ref_1;
    static std::string output;
    int failCount=0;
    int goodCount=0;
    int i;

#if 0
#ifdef __WIN32__
      ai.Open("c:\\Users\\vuong\\VM\\ztachip\\SW\\gguf\\LLAMA.ZUF");
#else
      ai.Open("LLAMA.ZUF");
#endif
#endif

    printf("\r\n Loading model from SD card \r\n");
    printf("\r\n Loading is slow since SPI driver for SD card is not hardware based yet.\r\n");
#if 1
#ifdef __WIN32__
      ai.Open("c:\\Users\\vuong\\VM\\ztachip\\SW\\gguf\\xxx.ZUF");
#else
      ai.Open("SMOLLM2.ZUF");
#endif
#endif

      ai.SetSamplingPolicy(0.5,0.85,20); // temperature=0.8,threshold=0.9;top-k=20
//      ai.SetSamplingPolicyGreedy();
#ifdef __WIN32__
    ai.SystemPrompt((char*)"You are a helpful assistant.");
    for(i=0;;i++) {
        ai.Clear();

        output.clear();
        ai.UserPrompt((char*)"Who is Issac Newton", &output);

        if(i==0)
            output_ref_0 = output;
        if (output_ref_0 != output) {
            failCount++;
        }
        else {
            goodCount++;
        }
        printf("\r\n--> SUCESS fail=%d good=%d \r\n",failCount,goodCount);
    }
    ai.Close();
#endif
#ifndef __WIN32__
    printf("I am a chatbot. Hit Ctrl+C to interrupt me.\r\n");
    ai.SystemPrompt((char*)"You are a helpful assistance");
    for(;;) {
        char *prompt = getInput();
        if(prompt) {
            if(prompt[0]==0x3)
                ai.Clear();
            else {
                // Since this is a small model. It does not handle long context well.
                // Clear previous context before answering new query
                ai.Clear(); 
                ai.ClearStat();
                ai.UserPrompt(prompt,0);
                printf(" (tok/sec=%.2f)",ai.GetStatTokPerSec());
            }
        }
    }
#endif
    return 0;
}
