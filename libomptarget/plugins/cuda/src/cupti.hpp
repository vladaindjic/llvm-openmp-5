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
#define __CUPTI_HPP__



//******************************************************************************
// include files
//******************************************************************************

#include <cupti_activity.h>



//******************************************************************************
// types
//******************************************************************************

typedef void (*cupti_correlation_callback_t)
(
 uint64_t *id
);


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
  cupti_load_callback_t load_callback,
  cupti_load_callback_t unload_callback
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
 CUcontext context,
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
cupti_trace_init
(
  CUpti_BuffersCallbackRequestFunc buffer_request, 
  CUpti_BuffersCallbackCompleteFunc buffer_complete
);


extern void
cupti_trace_flush
(
 CUcontext context
);


extern bool 
cupti_trace_start
(
 CUcontext context
);

extern bool 
cupti_trace_pause
(
 CUcontext context
);

extern bool 
cupti_trace_stop
(
 CUcontext context
);


//===----------------------------------------------------------------------===//
#endif // __CUPTI_HPP__
