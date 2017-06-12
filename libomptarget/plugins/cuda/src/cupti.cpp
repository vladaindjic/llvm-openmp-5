//===----RTLs/cuda/src/cupti.cpp----------------------------------- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//
//
// CUPTI interface for NVIDIA GPU
//
//===----------------------------------------------------------------------===//


//******************************************************************************
// system include files 
//******************************************************************************

#include <iostream>

#include <stdio.h>
#include <stdlib.h>



//******************************************************************************
// local include files
//******************************************************************************

#undef DEBUGP
#define DEBUGP(prefix, ...)                                                    \
  {                                                                            \
    fprintf(stderr, "%s --> ", prefix);                                        \
    fprintf(stderr, __VA_ARGS__);                                              \
  }

#include <inttypes.h>
#define DPxMOD "0x%0*" PRIxPTR
#define DPxPTR(ptr) ((int)(2*sizeof(uintptr_t))), ((uintptr_t) (ptr))

#include "rtl.h" 

#undef DP
#define DP(...)



//******************************************************************************
// cuda include 
//******************************************************************************

#include <cupti.h> 

#include "cupti.hpp" 
#include "cuda.hpp" 

  

//******************************************************************************
// macros
//******************************************************************************

#define CUPTI_ACTIVITY_BUFFER_SIZE (64 * 1024)


#define CUPTI_ACTIVITY_BUFFER_ALIGNMENT (8)

#define CUPTI_CALL(fn, args)			      \
  {						      \
    CUptiResult status = fn args;		      \
    if (status != CUPTI_SUCCESS) {		      \
      cupti_error_report(status, #fn);		      \
    }						      \
}

#define DISPATCH_CALLBACK(fn, args) if (fn) fn args


//******************************************************************************
// types
//******************************************************************************


typedef void (*cupti_error_callback_t) 
(
 const char *type, 
 const char *fn, 
 const char *error_string
);


typedef void (*cupti_dropped_callback_t) 
(
 size_t dropped
);


typedef CUptiResult (*cupti_activity_enable_disable_t) 
(
 CUcontext context,
 CUpti_ActivityKind activity
);


typedef struct {
  CUpti_BuffersCallbackRequestFunc buffer_request; 
  CUpti_BuffersCallbackCompleteFunc buffer_complete;
} cupti_activity_buffer_state_t;



//******************************************************************************
// forward declarations 
//******************************************************************************

static void
cupti_error_callback_dummy
(
 const char *type, 
 const char *fn, 
 const char *error_string
);


static void
cupti_dropped_callback_dummy
(
 size_t dropped
);


static void 
cupti_correlation_callback_dummy
(
 uint64_t *id
);



//******************************************************************************
// constants
//******************************************************************************

CUpti_ActivityKind
external_correlation_activities[] = {
  CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION, 
  CUPTI_ACTIVITY_KIND_INVALID
};


CUpti_ActivityKind
data_motion_explicit_activities[] = {
  CUPTI_ACTIVITY_KIND_MEMCPY, 
  CUPTI_ACTIVITY_KIND_INVALID
};


CUpti_ActivityKind
data_motion_implicit_activities[] = {
  CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER,
  CUPTI_ACTIVITY_KIND_MEMCPY2,
  CUPTI_ACTIVITY_KIND_INVALID
};


CUpti_ActivityKind
kernel_invocation_activities[] = {
  CUPTI_ACTIVITY_KIND_KERNEL,
  CUPTI_ACTIVITY_KIND_INVALID
};


CUpti_ActivityKind
kernel_execution_activities[] = {
  CUPTI_ACTIVITY_KIND_PC_SAMPLING,
  CUPTI_ACTIVITY_KIND_FUNCTION,
  CUPTI_ACTIVITY_KIND_INVALID
};


CUpti_ActivityKind
overhead_activities[] = {
  CUPTI_ACTIVITY_KIND_OVERHEAD,
  CUPTI_ACTIVITY_KIND_INVALID
};


CUpti_ActivityKind
driver_activities[] = {
  CUPTI_ACTIVITY_KIND_DRIVER,
  CUPTI_ACTIVITY_KIND_INVALID
};


CUpti_ActivityKind
runtime_activities[] = {
  CUPTI_ACTIVITY_KIND_RUNTIME,
  CUPTI_ACTIVITY_KIND_INVALID
};



//******************************************************************************
// static data
//******************************************************************************

cupti_correlation_callback_t cupti_correlation_callback = 
  cupti_correlation_callback_dummy;

static cupti_error_callback_t cupti_error_callback = 
  cupti_error_callback_dummy;

cupti_activity_buffer_state_t cupti_activity_enabled = { 0, 0 };
cupti_activity_buffer_state_t cupti_activity_disabled = { 0, 0 };

cupti_activity_buffer_state_t *cupti_activity_state = 
  &cupti_activity_disabled;

cupti_load_callback_t cupti_load_callback = 0;

cupti_load_callback_t cupti_unload_callback = 0;

static cupti_dropped_callback_t cupti_dropped_callback = 
  cupti_dropped_callback_dummy;
CUpti_SubscriberHandle cupti_subscriber;



//******************************************************************************
// internal functions
//******************************************************************************

static void
cupti_subscriber_callback
(
 void *userdata,
 CUpti_CallbackDomain domain,
 CUpti_CallbackId cb_id,
 const CUpti_CallbackData *cb_info
)
{
  DP("enter cupti_subscriber_callback\n");

  if (domain == CUPTI_CB_DOMAIN_RESOURCE) {
    const CUpti_ResourceData *rd = (const CUpti_ResourceData *) cb_info;
    if (cb_id == CUPTI_CBID_RESOURCE_MODULE_LOADED) {
      CUpti_ModuleResourceData *mrd = (CUpti_ModuleResourceData *) rd->resourceDescriptor;
      DP("loaded module id %d, cubin size %ld, cubin %p\n", 
	     mrd->moduleId, mrd->cubinSize, mrd->pCubin);
      DISPATCH_CALLBACK(cupti_load_callback, (mrd->moduleId, mrd->pCubin, mrd->cubinSize));
    }
    if (cb_id == CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING) {
      CUpti_ModuleResourceData *mrd = (CUpti_ModuleResourceData *) rd->resourceDescriptor;
      DP("unloaded module id %d, cubin size %ld, cubin %p\n", 
	     mrd->moduleId, mrd->cubinSize, mrd->pCubin);
      DISPATCH_CALLBACK(cupti_unload_callback, (mrd->moduleId, mrd->pCubin, mrd->cubinSize));
    }
  } else if (CUPTI_CB_DOMAIN_DRIVER_API) {
    if ((cb_id == CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2) ||
	(cb_id == CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2) ||
	(cb_id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel)){

      uint64_t correlation_id;
      DISPATCH_CALLBACK(cupti_correlation_callback, (&correlation_id));

      if (correlation_id != 0) {
	if (cb_info->callbackSite == CUPTI_API_ENTER) {
	  cuptiActivityPushExternalCorrelationId
	    (CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, correlation_id);
	}
	if (cb_info->callbackSite == CUPTI_API_EXIT) {
	  cuptiActivityPopExternalCorrelationId
	    (CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, &correlation_id);
	}
      }
    }
  }

  DP("exit cupti_subscriber_callback\n");
}


bool
cupti_device_get_timestamp
(
 CUcontext context,
 uint64_t *time
)
{
  uint64_t timestamp;

  CUptiResult get_result = cuptiDeviceGetTimestamp(context, &timestamp);

  bool time_result = (get_result == CUPTI_SUCCESS);

  if (time_result) {
    *time = timestamp;
  }

  return time_result;
}

static void 
cupti_correlation_callback_dummy // __attribute__((unused))
(
 uint64_t *id
)
{
  *id = 0;
}


void 
cupti_buffer_alloc 
(
 uint8_t **buffer, 
 size_t *buffer_size, 
 size_t *maxNumRecords
)
{
  int retval = posix_memalign((void **) buffer, 
			      (size_t) CUPTI_ACTIVITY_BUFFER_ALIGNMENT,
			      (size_t) CUPTI_ACTIVITY_BUFFER_SIZE); 
  
  if (retval != 0) {
    cupti_error_callback("CUPTI", "cupti_buffer_alloc", "out of memory");
  }
  
  *buffer_size = CUPTI_ACTIVITY_BUFFER_SIZE;

  *maxNumRecords = 0;
}



//******************************************************************************
// private operations
//******************************************************************************

static void
cupti_error_callback_dummy // __attribute__((unused))
(
 const char *type, 
 const char *fn, 
 const char *error_string
)
{
  std::cerr << type << ": function " << fn 
	    << " failed with error " << error_string << std::endl;                       
  exit(-1);
} 


static void
cupti_dropped_callback_dummy
(
 size_t dropped
)
{
  std::cerr << "CUPTI dropped " << dropped << " samples." << std::endl;
}


static void
cupti_error_report
(
 CUptiResult error, 
 const char *fn
)
{
  const char *error_string;
  cuptiGetResultString(error, &error_string);
  cupti_error_callback("CUPTI result error", fn, error_string);
} 

static bool
cupti_trace_restart
(
 CUcontext context,
 cupti_activity_buffer_state_t * cupti_activity_next_state
)
{
  cupti_activity_state = cupti_activity_next_state;

  CUptiResult cupti_result = cuptiActivityRegisterCallbacks
    (cupti_activity_state->buffer_request, cupti_activity_state->buffer_complete); 

  return (cupti_result == CUPTI_SUCCESS);
}



//******************************************************************************
// interface  operations
//******************************************************************************

//-------------------------------------------------------------
// event specification
//-------------------------------------------------------------

cupti_set_status_t
cupti_set_monitoring
(
 CUcontext context,
 const  CUpti_ActivityKind activity_kinds[],
 bool enable
)
{
  int failed = 0;
  int succeeded = 0;
  cupti_activity_enable_disable_t action =
    (enable ? cuptiActivityEnableContext : cuptiActivityDisableContext);	
  int i = 0;
  for (;;) {
    CUpti_ActivityKind activity_kind = activity_kinds[i++];
    if (activity_kind == CUPTI_ACTIVITY_KIND_INVALID) break;
    CUptiResult status = action(context, activity_kind);
    if (status == CUPTI_SUCCESS) succeeded++;
    else failed++;
  }
  if (succeeded > 0) {
    if (failed == 0) return cupti_set_all;
    else return cupti_set_some;
  }
  return cupti_set_none;
}


//-------------------------------------------------------------
// tracing control 
//-------------------------------------------------------------

void 
cupti_trace_init
(
  CUpti_BuffersCallbackRequestFunc buffer_request, 
  CUpti_BuffersCallbackCompleteFunc buffer_complete
)
{
  cupti_activity_enabled.buffer_request = buffer_request;
  cupti_activity_enabled.buffer_complete = buffer_complete;
}


void
cupti_trace_flush
(
 CUcontext context
)
{
  CUPTI_CALL(cuptiActivityFlushAll, (CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
}


bool 
cupti_trace_start
(
 CUcontext context
)
{
  return cupti_trace_restart(context, &cupti_activity_enabled);
}


bool 
cupti_trace_pause
(
 CUcontext context
)
{
  cupti_trace_flush(context);
  return cupti_trace_restart(context, &cupti_activity_disabled);
}


bool 
cupti_trace_stop
(
 CUcontext context
)
{
  return cupti_trace_pause(context);
}


//-------------------------------------------------------------
// correlation callback control 
//-------------------------------------------------------------

void
cupti_correlation_enable
(
  cupti_load_callback_t load_callback,
  cupti_load_callback_t unload_callback
)
{
  cupti_load_callback = load_callback;
  cupti_unload_callback = unload_callback;

  if (cupti_correlation_callback) {
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION);

    cuptiSubscribe(&cupti_subscriber, 
		   (CUpti_CallbackFunc) cupti_subscriber_callback,
		   (void *) NULL);

    cuptiEnableDomain(1, cupti_subscriber, CUPTI_CB_DOMAIN_DRIVER_API);
    cuptiEnableDomain(1, cupti_subscriber, CUPTI_CB_DOMAIN_RESOURCE);
  }
}


void
cupti_correlation_disable()
{
  cuptiActivityDisable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION);

  cuptiUnsubscribe(cupti_subscriber); 

  cuptiEnableDomain(0, cupti_subscriber, CUPTI_CB_DOMAIN_DRIVER_API);
  cuptiEnableDomain(0, cupti_subscriber, CUPTI_CB_DOMAIN_RESOURCE);

  cupti_load_callback = 0;
  cupti_unload_callback = 0;
}


void
cupti_correlation_callback_register
(
 cupti_correlation_callback_t callback_fn
)
{
  cupti_correlation_callback = callback_fn;
}


//-------------------------------------------------------------
// cursor support
//-------------------------------------------------------------
  
bool
cupti_buffer_cursor_advance
(
 uint8_t *buffer,
 size_t size,
 CUpti_Activity **activity
)
{
  bool status;
  CUptiResult result = cuptiActivityGetNextRecord(buffer, size, activity);
  status = (result == CUPTI_SUCCESS);
  return status;
}


bool
cupti_buffer_cursor_isvalid
(
 uint8_t *buffer,
 size_t size,
 CUpti_Activity *activity
)
{
  CUpti_Activity *cursor = activity;
  return cupti_buffer_cursor_advance(buffer, size, &cursor);
}

