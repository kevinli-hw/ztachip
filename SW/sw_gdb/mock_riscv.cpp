extern "C" {
  char _heap_end = 0;
  char _heap_start = 0;
  int heap_avail() { return 0; }
  int heap_usage() { return 0; }
  void* _sbrk(int) { return nullptr; }
  void UartWriteAvailable() {}
  void UartWrite(const char*, unsigned long) {}
  void _taskYield() {}
  void _taskSpawn(unsigned int (*)(void*, int), void*, int) {}
//   void ztaDualHartExecute(void (*)(void*, int), void*) {}
  int kernel_add_exe() { return 0; }
  int kernel_concatenate_exe() { return 0; }
  int kernel_innerProduct_exe() { return 0; }
  int kernel_convolution_exe() { return 0; }
  int kernel_convolution_depthwise_exe() { return 0; }
  int kernel_logistic_exe() { return 0; }
  int kernel_objdet_exe() { return 0; }
  int kernel_Pooling_exe() { return 0; }

}
