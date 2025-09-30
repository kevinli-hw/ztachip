# Makefile (flat obj dir)
TARGET_NAME  	:= test_model_exec
CXX     			:= g++
CC      			:= gcc

OBJDIR  := build

INCLUDES  := -I../base -I../apps/nn/flatbuffer
CXXFLAGS  := -g -fpermissive $(INCLUDES) -MMD -MP
CFLAGS    := -g $(INCLUDES) -MMD -MP

C_SRCS := \
  mini_newlib.c \
  ../base/ztalib.c \
  ../base/util.c \
	../fs/gen/conv7x7_fc120_int8.c \
	../fs/gen/conv7x7_conv_int8.c \
	../fs/gen/mobilenet_v2_1_0_224_quant.c \
	../fs/gen/resnet50_int8.c

CXX_SRCS := \
  example.cpp \
  mock_riscv.cpp \
  ../base/tensor.cpp \
  ../base/graph.cpp \
  ../apps/nn/nn_add.cpp \
  ../apps/nn/nn_concat.cpp \
  ../apps/nn/nn_conv2d.cpp \
  ../apps/nn/nn.cpp \
  ../apps/nn/nn_logistic.cpp \
  ../apps/nn/nn_objdect.cpp \
  ../apps/nn/nn_poolavg.cpp \
  ../apps/nn/nn_reshape.cpp \
  ../apps/nn/nn_util.cpp \
  ../apps/nn/tf.cpp \
  ../apps/nn/tf_util.cpp

TARGET 	 := $(OBJDIR)/$(TARGET_NAME) 

C_OBJS   := $(addprefix $(OBJDIR)/,$(notdir $(C_SRCS:.c=.o)))
CXX_OBJS := $(addprefix $(OBJDIR)/,$(notdir $(CXX_SRCS:.cpp=.o)))
OBJS     := $(C_OBJS) $(CXX_OBJS)
DEPS     := $(OBJS:.o=.d)

VPATH := . ../base ../apps/nn ../apps/nn/.. ../fs/gen

all: $(TARGET)

$(TARGET): $(OBJS) | $(OBJDIR)
	$(CXX) $(OBJS) -o $@

$(OBJDIR):
	@mkdir -p $@

$(OBJDIR)/%.o: %.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/%.o: %.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJDIR) $(TARGET)

-include $(DEPS)

