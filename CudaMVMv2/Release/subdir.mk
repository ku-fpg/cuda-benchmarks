################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../MVMv2.cu 

CU_DEPS += \
./MVMv2.d 

OBJS += \
./MVMv2.o 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/opt/cuda/bin/nvcc -O3 -gencode arch=compute_11,code=sm_11 -gencode arch=compute_35,code=sm_35 -odir "" -M -o "$(@:%.o=%.d)" "$<"
	/opt/cuda/bin/nvcc --compile -O3 -gencode arch=compute_11,code=compute_11 -gencode arch=compute_11,code=sm_11 -gencode arch=compute_35,code=sm_35  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


