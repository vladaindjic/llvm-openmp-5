//===----RTLs/cuda/src/cupti.hpp----------------------------------- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//
//
// CUPTI interface for NVIDIA acclerator
//
//===----------------------------------------------------------------------===//
#ifndef __CUPTI_HPP__

//******************************************************************************
// include files
//******************************************************************************

#include <cupti_activity.h>



//******************************************************************************
// macros
//******************************************************************************

#define FOREACH_ACTIVITY_KIND(macro)	\
  macro(INVALID)			\
  macro(MEMCPY)				\
  macro(MEMSET)				\
  macro(KERNEL)				\
  macro(DRIVER)				\
  macro(RUNTIME)			\
  macro(EVENT)				\
  macro(METRIC)				\
  macro(DEVICE)				\
  macro(CONTEXT)			\
  macro(CONCURRENT_KERNEL)		\
  macro(NAME)				\
  macro(MARKER)				\
  macro(MARKER_DATA)			\
  macro(SOURCE_LOCATOR)			\
  macro(GLOBAL_ACCESS)			\
  macro(BRANCH)				\
  macro(OVERHEAD)			\
  macro(CDP_KERNEL)			\
  macro(PREEMPTION)			\
  macro(ENVIRONMENT)			\
  macro(EVENT_INSTANCE)			\
  macro(MEMCPY2)			\
  macro(METRIC_INSTANCE)		\
  macro(INSTRUCTION_EXECUTION)		\
  macro(UNIFIED_MEMORY_COUNTER)		\
  macro(FUNCTION)			\
  macro(MODULE)				\
  macro(DEVICE_ATTRIBUTE)		\
  macro(SHARED_ACCESS)			\
  macro(PC_SAMPLING)			\
  macro(PC_SAMPLING_RECORD_INFO)	\
  macro(INSTRUCTION_CORRELATION)	\
  macro(OPENACC_DATA)			\
  macro(OPENACC_LAUNCH)			\
  macro(OPENACC_OTHER)			\
  macro(CUDA_EVENT)			\
  macro(STREAM)				\
  macro(SYNCHRONIZATION)		\
  macro(EXTERNAL_CORRELATION)		\
  macro(NVLINK)				\
  macro(INSTANTANEOUS_EVENT)		\
  macro(INSTANTANEOUS_EVENT_INSTANCE)	\
  macro(INSTANTANEOUS_METRIC)		\
  macro(INSTANTANEOUS_METRIC_INSTANCE)	\
  macro(FORCE_INT)


//******************************************************************************
// types
//******************************************************************************

typedef void (*cupti_correlation_callback_t)
(
 uint64_t *id
);


typedef void (*cupti_activity_fn_t) 
(
 CUpti_Activity *activity,
 void *state
);


typedef struct {
#define macro(kind) cupti_activity_fn_t kind;
  FOREACH_ACTIVITY_KIND(macro)
#undef macro
} cupti_activity_dispatch_t;


typedef void (*cupti_load_callback_t)
(
 int module_id, 
 const void *cubin, 
 size_t cubin_size
);



//******************************************************************************
// constants
//******************************************************************************

extern CUpti_ActivityKind
external_correlation_activities[];

extern CUpti_ActivityKind
data_motion_explicit_activities[];

extern CUpti_ActivityKind
data_motion_implicit_activities[];

extern CUpti_ActivityKind
kernel_invocation_activities[];

extern CUpti_ActivityKind
kernel_execution_activities[];

extern CUpti_ActivityKind
driver_activities[];

extern CUpti_ActivityKind
runtime_activities[];

extern CUpti_ActivityKind
overhead_activities[];

typedef enum {
  cupti_set_all = 1,
  cupti_set_some = 2,
  cupti_set_none = 3
} cupti_set_status_t;



//******************************************************************************
// interface functions
//******************************************************************************

extern void 
cupti_buffer_alloc 
(
 uint8_t **buffer, 
 size_t *buffer_size, 
 size_t *maxNumRecords
);


extern bool
cupti_buffer_cursor_advance
(
 uint8_t *buffer,
 size_t validSize,
 CUpti_Activity **activity
);


extern bool
cupti_buffer_cursor_isvalid
(
 uint8_t *buffer,
 size_t validSize,
 CUpti_Activity *activity
);

void
cupti_correlation_enable
(
);

extern void
cupti_correlation_disable
(
);

extern void
cupti_correlation_callback_register
(
 cupti_correlation_callback_t callback_fn
);


extern void
cupti_activity_buffer_mgmt_init
(
  CUpti_BuffersCallbackRequestFunc buffer_request, 
  CUpti_BuffersCallbackCompleteFunc buffer_complete,
  void *buffer_processing_state
);


extern bool
cupti_pause_trace
(
 CUcontext context,
 int begin_pause
);


extern cupti_set_status_t 
cupti_set_monitoring
(
 const  CUpti_ActivityKind activity_kinds[],
 bool enable
);


extern bool
cupti_device_get_timestamp
(
 CUcontext context,
 uint64_t *time
);


extern void
cupti_stop
(
);


//===----------------------------------------------------------------------===//
#endif // __CUPTI_HPP__
