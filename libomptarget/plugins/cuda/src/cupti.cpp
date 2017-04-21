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

#if 1
#include <stdio.h>
#include <stdlib.h>
#endif

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


#define FOREACH_CUPTI_STATUS(macro)	              \
  macro(CUPTI_SUCCESS)				      \
  macro(CUPTI_ERROR_INVALID_PARAMETER)		      \
  macro(CUPTI_ERROR_INVALID_DEVICE)		      \
  macro(CUPTI_ERROR_INVALID_CONTEXT)		      \
  macro(CUPTI_ERROR_NOT_INITIALIZED) 	


#define FOREACH_ACTIVITY_OVERHEAD(macro)	      \
  macro(DRIVER, COMPILER)			      \
  macro(CUPTI,  BUFFER_FLUSH)			      \
  macro(CUPTI,  INSTRUMENTATION)		      \
  macro(CUPTI,  RESOURCE)


#define FOREACH_OBJECT_KIND(macro)	              \
  macro(PROCESS, pt.processId)			      \
  macro(THREAD,  pt.threadId)			      \
  macro(DEVICE,  dcs.deviceId)			      \
  macro(CONTEXT, dcs.contextId)			      \
  macro(STREAM,  dcs.streamId)


#define FOREACH_MEMCPY_KIND(macro)                    \
  macro(ATOA)					      \
  macro(ATOD)					      \
  macro(ATOH)					      \
  macro(DTOA)					      \
  macro(DTOD)					      \
  macro(DTOH)					      \
  macro(HTOA)					      \
  macro(HTOD)					      \
  macro(HTOH)


#define DISPATCH_CALLBACK(fn, args) if (fn) fn args


#define FOREACH_STALL_REASON(macro)                   \
  macro(INVALID)				      \
  macro(NONE)					      \
  macro(INST_FETCH)				      \
  macro(EXEC_DEPENDENCY)			      \
  macro(MEMORY_DEPENDENCY)			      \
  macro(TEXTURE)				      \
  macro(SYNC)					      \
  macro(CONSTANT_MEMORY_DEPENDENCY)		      \
  macro(PIPE_BUSY)				      \
  macro(MEMORY_THROTTLE)			      \
  macro(NOT_SELECTED)				      \
  macro(OTHER)


#define CUPTI_CALL(fn, args)			      \
  {						      \
    CUptiResult status = fn args;		      \
    if (status != CUPTI_SUCCESS) {		      \
      cupti_error_report(status, #fn);		      \
    }						      \
}



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

cupti_activity_dispatch_t cupti_activity_dispatch_print;

cupti_activity_dispatch_t *cupti_activity_dispatch = 
  &cupti_activity_dispatch_print;

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


#define gpu_event_decl(stall) \
  int gpu_stall_event_ ## stall; 

FOREACH_STALL_REASON(gpu_event_decl)

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


const char*
cupti_status_to_string
(
 uint32_t err
)
{
#define CUPTI_STATUS_TO_STRING(s) if (err == s) return #s;
  
  FOREACH_CUPTI_STATUS(CUPTI_STATUS_TO_STRING);

#undef CUPTI_STATUS_TO_STRING			
  
  return "CUPTI_STATUS_UNKNOWN";
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


cupti_set_status_t
cupti_set_monitoring
(
 const  CUpti_ActivityKind activity_kinds[],
 bool enable
)
{
  int failed = 0;
  int succeeded = 0;
  cupti_activity_enable_disable_t action =
    (enable ? cuptiActivityEnable : cuptiActivityDisable);	
  int i = 0;
  for (;;) {
    CUpti_ActivityKind activity_kind = activity_kinds[i++];
    if (activity_kind == CUPTI_ACTIVITY_KIND_INVALID) break;
    CUptiResult status = action(activity_kind);
    if (status == CUPTI_SUCCESS) succeeded++;
    else failed++;
  }
  if (succeeded > 0) {
    if (failed == 0) return cupti_set_all;
    else return cupti_set_some;
  }
  return cupti_set_none;
}

static bool
cupti_trace_restart
(
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
// tracing control 
//-------------------------------------------------------------

bool cupti_trace_init
(
  CUpti_BuffersCallbackRequestFunc buffer_request, 
  CUpti_BuffersCallbackCompleteFunc buffer_complete
)
{
  cupti_activity_enabled.buffer_request = buffer_request;
  cupti_activity_enabled.buffer_complete = buffer_complete;
}


void
cupti_trace_flush()
{
  CUPTI_CALL(cuptiActivityFlushAll, (0));
}


bool 
cupti_trace_start
(
)
{
  return cupti_trace_restart(&cupti_activity_enabled);
}


bool 
cupti_trace_pause
(
)
{
  cupti_trace_flush();
  return cupti_trace_restart(&cupti_activity_disabled);
}


bool 
cupti_trace_stop
(
)
{
  return cupti_trace_pause();
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


//-------------------------------------------------------------
// printing support
//-------------------------------------------------------------

const char *
cupti_activity_overhead_kind_string
(
 CUpti_ActivityOverheadKind kind
)
{
#define macro(oclass, otype)						\
  case CUPTI_ACTIVITY_OVERHEAD_ ## oclass ## _ ## otype: return #otype;

  switch(kind) {
    FOREACH_ACTIVITY_OVERHEAD(macro)
  default: return "UNKNOWN";
  }

#undef macro
}


const char *
cupti_memcpy_kind_string
(
 CUpti_ActivityMemcpyKind kind
)
{
#define macro(name)						\
  case CUPTI_ACTIVITY_MEMCPY_KIND_ ## name: return #name;

  switch(kind) {
    FOREACH_MEMCPY_KIND(macro)
  default: return "UNKNOWN";
  }

#undef macro
}


const char *
cupti_stall_reason_string
(
 CUpti_ActivityPCSamplingStallReason kind
)
{
#define macro(stall)							\
  case CUPTI_ACTIVITY_PC_SAMPLING_STALL_ ## stall: return #stall; 
  
  switch (kind) {
    FOREACH_STALL_REASON(macro)
  default: return "UNKNOWN";
  }

#undef macro 
}


const char *
cupti_activity_object_kind_string
(
 CUpti_ActivityObjectKind kind
)
{
#define macro(name, id)					\
  case CUPTI_ACTIVITY_OBJECT_ ## name: return #name;

  switch(kind) {
    FOREACH_OBJECT_KIND(macro)
  default: return "UNKNOWN";
  }

#undef macro
}


uint32_t 
cupti_activity_object_kind_id
(
 CUpti_ActivityObjectKind kind, 
 CUpti_ActivityObjectKindId *kid 
)
{
#define macro(name, id)					\
  case CUPTI_ACTIVITY_OBJECT_ ## name: return kid->id;

  switch(kind) {
    FOREACH_OBJECT_KIND(macro)
  default: return -1;
  }

#undef macro
}

