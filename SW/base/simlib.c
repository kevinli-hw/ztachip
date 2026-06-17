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
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include "../src/soc.h"
//#include "../fs/gen/single_conv_int8.c"
//#include "../fs/gen/single_fc_int8.c"
#include "../fs/gen/mean_model.c"
//#include "../fs/gen/mobilenet_v2_int8.c"
//#include "../fs/gen/classifier_input.c"
//#include "../fs/gen/classifier.c"
//#include "../fs/gen/mobilenet_v2_1_0_224_quant.c"

// This file implements functions required by newlib
// Functions implement filesystem calls, task management and memory management

#define FP_FIRST    (STDERR_FILENO+1)
#define FP_MAX_NUM  16

extern void _heap_start();
extern void _heap_end();

static unsigned int heap=(unsigned int)_heap_start;

// List of files opened

static struct {
   bool status;
   int len;
   int curr;
   const uint8_t *body;
} files[FP_MAX_NUM];

// Kill a process
// Not implemented...

int _kill(int pid, int sig) {
    errno = EINVAL;
    return -1;
}

// Get process id
// There is only one process

int _getpid(void) {
    return 1;
}

// Is file console output

int _isatty(int file) {
    return (file == STDOUT_FILENO || file == STDERR_FILENO);
}

// Exit a process
// Not implemented

void _exit(int code) {
    for(;;) {}
}

// Allocate memory block from heap

void *_sbrk (int nbytes) {
   void *p;
   p=(void *)heap;
   heap += nbytes;
   if(heap >= (unsigned int)_heap_end)
      _exit(-1);
   return p;
}

// Return amount of heap that get used

unsigned int heap_usage() {
    return (unsigned int)heap-(unsigned int)_heap_start;
}

unsigned int heap_avail() {
    return (unsigned int)_heap_end-(unsigned int)heap;
}

// Open a file
// There are some files pre-defined as C array

int _open(const char *name, int flags, int mode) {
   int i;
   for(i=0;i < FP_MAX_NUM;i++) {
      if(!files[i].status)
         break;
   }
   if(i >= FP_MAX_NUM) {
      errno = ENOENT;
      return -1;
   }
   //if(strcmp(name,"single_conv_int8.tflite")==0) {
	 //   files[i].status=true;
	 //   files[i].curr=0;
   //   files[i].len=sizeof(single_conv_int8);
   //   files[i].body=single_conv_int8;
   //}
   if(strcmp(name,"mean_model.tflite")==0) {
	    files[i].status=true;
	    files[i].curr=0;
      files[i].len=sizeof(mean_model);
      files[i].body=mean_model;
   }
   //if(strcmp(name,"single_fc_int8.tflite")==0) {
	 //   files[i].status=true;
	 //   files[i].curr=0;
   //   files[i].len=sizeof(single_fc_int8);
   //   files[i].body=single_fc_int8;
   //}
   //if(strcmp(name,"mobilenet_v2_int8.tflite")==0) {
	 //   files[i].status=true;
	 //   files[i].curr=0;
   //   files[i].len=sizeof(mobilenet_v2_int8);
   //   files[i].body=mobilenet_v2_int8;
   //}
   //else if(strcmp(name,"classifier_input.bmp")==0) {
	 // files[i].status=true;
	 // files[i].curr=0;
   //   files[i].len=sizeof(classifier_input);
   //   files[i].body=classifier_input;
   //}
   //else if(strcmp(name,"classifier.bin")==0) {
	 // files[i].status=true;
	 // files[i].curr=0;
   //   files[i].len=sizeof(classifier);
   //   files[i].body=classifier;
   //}
   //else if(strcmp(name,"mobilenet_v2_1_0_224_quant.tflite")==0) {
	 // files[i].status=true;
	 // files[i].curr=0;
   //   files[i].len=sizeof(mobilenet_v2_1_0_224_quant);
   //   files[i].body=mobilenet_v2_1_0_224_quant;
   //}
   else {
      errno = ENOENT;
      return -1;
   }
   return i+FP_FIRST;
   //errno = ENOENT;
   //return -1;
}

// Close a file

int _close(int file) {
   file -= FP_FIRST;
   if((file < 0 || file >= FP_MAX_NUM) || !files[file].status) {
      errno = EBADF;
      return -1;
   }
   files[file].status=false;
   return 0;
}

// Read from an opened file

ssize_t _read(int file, void *ptr, size_t len) {
   int remain;
   file -= FP_FIRST;
   if((file < 0 || file >= FP_MAX_NUM) || !files[file].status) {
      errno = EBADF;
      return -1;
   }
   remain=files[file].len-files[file].curr;
   if(len > remain)
      len=remain;
   memcpy(ptr,files[file].body+files[file].curr,len);
   files[file].curr += len;
   return len;
}

// Get statistics about the file such as its length

int _fstat(int file, struct stat *st) {
   file -= FP_FIRST;
   if((file < 0 || file >= FP_MAX_NUM) || !files[file].status) {
      errno = EBADF;
      return -1;
   }
   memset(st,0,sizeof(struct stat));
   st->st_size=files[file].len;
   return 0;
}

// Write to a file
// Not implemented

ssize_t _write(int fd, const void *ptr, size_t len) {
   if(fd == STDOUT_FILENO || fd == STDERR_FILENO) {
      int i;
      char *p;
      for(i=0,p=(char *)ptr;i < len;i++,p++) {
         while(UartWriteAvailable()==0);
         UartWrite(*p);
      }
      return len;
   }
   else {
      errno = ENOSYS;
      return -1;
   }
}

// Position a read cursor of an opened file

off_t _lseek(int file, off_t ptr, int dir) {
   file -= FP_FIRST;
   if((file < 0 || file >= FP_MAX_NUM) || !files[file].status) {
      errno = EBADF;
      return -1;
   }
   if(dir==SEEK_SET)
      files[file].curr=ptr;
   else if(dir==SEEK_CUR)
      files[file].curr+=ptr;
   else
      files[file].curr=files[file].len;
   if(files[file].curr>files[file].len)
      files[file].curr=files[file].len;
   return files[file].curr;
}
