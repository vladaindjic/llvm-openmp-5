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

#define FOREACH_CUPTI_STATUS(macro)	\
  macro(CUPTI_SUCCESS)			\
  macro(CUPTI_ERROR_INVALID_PARAMETER)	\
  macro(CUPTI_ERROR_INVALID_DEVICE)	\
  macro(CUPTI_ERROR_INVALID_CONTEXT)	\
  macro(CUPTI_ERROR_NOT_INITIALIZED) 	

#define FOREACH_ACTIVITY_OVERHEAD(macro)	\
  macro(DRIVER, COMPILER)			\
  macro(CUPTI, BUFFER_FLUSH)			\
  macro(CUPTI, INSTRUMENTATION)			\
  macro(CUPTI, RESOURCE)

#define FOREACH_OBJECT_KIND(macro)	\
  macro(PROCESS, pt.processId)			\
  macro(THREAD, pt.threadId)			\
  macro(DEVICE, dcs.deviceId)			\
  macro(CONTEXT, dcs.contextId)			\
  macro(STREAM, dcs.streamId)

#define FOREACH_MEMCPY_KIND(macro) \
  macro(ATOA)			   \
  macro(ATOD)			   \
  macro(ATOH)			   \
  macro(DTOA)			   \
  macro(DTOD)			   \
  macro(DTOH)			   \
  macro(HTOA)			   \
  macro(HTOD)			   \
  macro(HTOH)

#define DISPATCH_CALLBACK(fn, args) if (fn) fn args


#define FOREACH_STALL_REASON(macro)             \
  macro(INVALID)				\
  macro(NONE)					\
  macro(INST_FETCH)				\
  macro(EXEC_DEPENDENCY)			\
  macro(MEMORY_DEPENDENCY)			\
  macro(TEXTURE)				\
  macro(SYNC)					\
  macro(CONSTANT_MEMORY_DEPENDENCY)		\
  macro(PIPE_BUSY)				\
  macro(MEMORY_THROTTLE)			\
  macro(NOT_SELECTED)				\
  macro(OTHER)


#define CUPTI_CALL(fn, args)						\
  {									\
    CUptiResult status = fn args;                                       \
    if (status != CUPTI_SUCCESS) {                                      \
      cupti_error_report(status, #fn);					\
    }                                                                   \
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
  void *buffer_processing_state;
} cupti_activity_buffer_mgmt_t;



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


static void 
cupti_error_callback_dummy
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

cupti_correlation_callback_t 
cupti_correlation_callback = 0;

cupti_activity_dispatch_t cupti_activity_dispatch_print;

cupti_activity_dispatch_t *
cupti_activity_dispatch = &cupti_activity_dispatch_print;

static cupti_error_callback_t 
cupti_error_callback = cupti_error_callback_dummy;

cupti_activity_buffer_mgmt_t cupti_activity_buffer_mgmt_data;

cupti_activity_buffer_mgmt_t *cupti_activity_buffer_mgmt = 0;



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
  if (domain == CUPTI_CB_DOMAIN_RESOURCE) {
    CUpti_ResourceData *rd = (CUpti_ResourceData *) cb_info;
    if (cb_id == CUPTI_CBID_RESOURCE_MODULE_LOADED) {
      CUpti_ModuleResourceData *mrd = (CUpti_ModuleResourceData *) rd->resourceDescriptor;
      printf("loaded module id %d, cubin size %ld, cubin %p\n", 
	     mrd->moduleId, mrd->cubinSize, mrd->pCubin);
    }
    if (cb_id == CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING) {
      CUpti_ModuleResourceData *mrd = (CUpti_ModuleResourceData *) rd->resourceDescriptor;
      printf("unloaded module id %d, cubin size %ld, cubin %p\n", 
	     mrd->moduleId, mrd->cubinSize, mrd->pCubin);
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
cupti_device_get_time
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
cupti_correlation_callback_dummy
(
 uint64_t *id
)
{
  id = 0;
}


static void 
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
// interface functions
//******************************************************************************

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


static const char *
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

  
bool
cupti_buffer_cursor_advance
(
 uint8_t *buffer,
 size_t validSize,
 CUpti_Activity **activity
)
{
  bool status;
  CUptiResult result = cuptiActivityGetNextRecord(buffer, validSize, activity);
  return (result == CUPTI_SUCCESS);
}


bool
cupti_buffer_cursor_isvalid
(
 uint8_t *buffer,
 size_t validSize,
 CUpti_Activity *activity
)
{
  CUpti_Activity *cursor = activity;
  return cupti_buffer_cursor_advance(buffer, validSize, &cursor);
}


void
cupti_correlation_callback_register
(
 cupti_correlation_callback_t callback_fn
)
{
  cupti_correlation_callback = callback_fn;
}


void
cupti_activity_buffer_mgmt_init
(
  CUpti_BuffersCallbackRequestFunc buffer_request, 
  CUpti_BuffersCallbackCompleteFunc buffer_complete,
  void *buffer_processing_state
)
{
  cupti_activity_buffer_mgmt_t *cabm = &cupti_activity_buffer_mgmt_data;
#define ASSIGN_FIELD(f) cabm->f = f
  ASSIGN_FIELD(buffer_request);
  ASSIGN_FIELD(buffer_complete);
  ASSIGN_FIELD(buffer_processing_state);
#undef ASSIGN_FIELD
  cupti_activity_buffer_mgmt = &cupti_activity_buffer_mgmt_data;
}


bool
cupti_pause_trace
(
 CUcontext context,
 int begin_pause
)
{
  bool result = false;

  if (cupti_activity_buffer_mgmt) {
    if (cuda_context_set(context)) {
      if (begin_pause) {
      } else {
	CUptiResult cupti_result =
	  cuptiActivityRegisterCallbacks
	  (cupti_activity_buffer_mgmt->buffer_request, 
	   cupti_activity_buffer_mgmt->buffer_complete); 

	result = (cupti_result == CUPTI_SUCCESS);
      }
    }
  }

  return result;
}


#define EMSG(...) 
#define monitor_real_abort() abort()
#define hpctoolkit_stats_increment(...)
#define get_correlation_id(ptr) (*(ptr) = 7) // test

static cupti_dropped_callback_t 
cupti_dropped_callback = cupti_dropped_callback_dummy;


#define gpu_event_decl(stall) \
  int gpu_stall_event_ ## stall; 

FOREACH_STALL_REASON(gpu_event_decl)

int illegal_event;

CUpti_SubscriberHandle cupti_subscriber;



//******************************************************************************
// private operations
//******************************************************************************

static void
cupti_error_callback_dummy
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
  bool result = true;
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


static void
cupti_note_dropped
(
 CUcontext ctx, 
 uint32_t stream_id
)
{
    size_t dropped;
    CUPTI_CALL(cuptiActivityGetNumDroppedRecords, (ctx, stream_id, &dropped));
    if (dropped != 0) {
      cupti_dropped_callback(dropped);
    }
}


static void
cupti_activity_process_unknown
(
 CUpti_Activity *activity,
 void *state
)    
{
  printf("Unknown activity kind %d\n", activity->kind);
}


static void
cupti_activity_print
(
 CUpti_Activity *activity,
 void *state
)
{
  const char *name;

#define macro(kind)							\
  case CUPTI_ACTIVITY_KIND_ ## kind: name = #kind; break;

  switch (activity->kind) {
    FOREACH_ACTIVITY_KIND(macro);

  default: name = "UNKNOWN"; break;
  }
#undef macro
  std::cout << "Activity " << name << std::endl;
}


void
cupti_activity_dispatch_print_init()
{
#define macro(kind) cupti_activity_dispatch_print.kind = cupti_activity_print;
  FOREACH_ACTIVITY_KIND(macro)
#undef macro
} 


static void
cupti_activity_process
(
 CUpti_Activity *activity,
 void *state
)
{
#define macro(kind)							\
  case CUPTI_ACTIVITY_KIND_ ## kind:					\
    cupti_activity_dispatch->kind(activity, state);			\
    break;
  
  switch (activity->kind) {
    FOREACH_ACTIVITY_KIND(macro);
  default:
    cupti_activity_process_unknown(activity, state);
    break;
  }
#undef macro
}


static void 
cupti_buffer_process_and_free
(
 CUcontext ctx, 
 uint32_t stream_id, 
 uint8_t *buffer, 
 size_t buffer_size, 
 size_t valid_size
)
{
  CUpti_Activity *activity = NULL;
  CUptiResult result;
  for(;;) {
    result = cuptiActivityGetNextRecord(buffer, valid_size, &activity);
    if (result == CUPTI_SUCCESS) {
      cupti_activity_process
	(activity, cupti_activity_buffer_mgmt->buffer_processing_state);
    } else {
      if (result != CUPTI_ERROR_MAX_LIMIT_REACHED) {
	cupti_error_report(result, "cuptiActivityGetNextRecord");
      }
      break;
    }
  }

  cupti_note_dropped(ctx, stream_id);

  free(buffer);
}


static void
cupti_correlation_enable()
{
  if (cupti_correlation_callback) {
    cuptiSubscribe(&cupti_subscriber, 
		   (CUpti_CallbackFunc) cupti_subscriber_callback,
		   (void *) NULL);
    cuptiEnableDomain(1, cupti_subscriber, CUPTI_CB_DOMAIN_DRIVER_API);
    cuptiEnableDomain(1, cupti_subscriber, CUPTI_CB_DOMAIN_RESOURCE);
  }
}


static void
cupti_correlation_disable()
{
  cuptiUnsubscribe(cupti_subscriber); 
  cuptiEnableDomain(0, cupti_subscriber, CUPTI_CB_DOMAIN_DRIVER_API);
  cuptiEnableDomain(0, cupti_subscriber, CUPTI_CB_DOMAIN_RESOURCE);
}


//******************************************************************************
// interface  operations
//******************************************************************************

void
cupti_start()
{
  cupti_activity_buffer_mgmt_init
    (cupti_buffer_alloc, cupti_buffer_process_and_free, 0);
  
  cupti_activity_dispatch_print_init();

  cupti_correlation_enable();
  cupti_set_monitoring(kernel_execution_activities, true);
  CUPTI_CALL(cuptiActivityRegisterCallbacks, 
	     (cupti_buffer_alloc, cupti_buffer_process_and_free));
}


void
cupti_stop()
{
  cupti_correlation_disable();
  CUPTI_CALL(cuptiActivityFlushAll, (0));
}



//******************************************************************************
// extra code  
//******************************************************************************

#if 0
static void
cupti_process_sample
(
 CUpti_ActivityPCSampling2 *sample,
 void *state
)
{
  printf("source %u, functionId %u, pc 0x%x, corr %u, "
	 "samples %u, stallreason %s\n",
	 sample->sourceLocatorId,
	 sample->functionId,
	 sample->pcOffset,
	 sample->correlationId,
	 sample->samples,
	 cupti_stall_reason_string(sample->stallReason));
}


static void
cupti_process_source_locator
(
 CUpti_ActivitySourceLocator *asl,
 void *state
)
{
  printf("Source Locator Id %d, File %s Line %d\n", 
	 asl->id, asl->fileName, 
	 asl->lineNumber);
}


static void
cupti_process_function
(
 CUpti_ActivityFunction *af,
 void *state
)
{
  printf("Function Id %u, ctx %u, moduleId %u, functionIndex %u, name %s\n",
	 af->id,
	 af->contextId,
	 af->moduleId,
	 af->functionIndex,
	 af->name);
}


static void
cupti_process_sampling_record_info
(
 CUpti_ActivityPCSamplingRecordInfo *sri,
 void *state
)
{
  printf("corr %u, totalSamples %llu, droppedSamples %llu\n",
	 sri->correlationId,
	 (unsigned long long)sri->totalSamples,
	 (unsigned long long)sri->droppedSamples);
}


static void
cupti_process_correlation
(
 CUpti_ActivityExternalCorrelation *ec,
 void *state
)
{
  state->external_correlation_id = ec->externalId;
  printf("External CorrelationId %llu\n", ec->externalId);
}


static void
cupti_process_memcpy
(
 CUpti_ActivityMemcpy *activity,
 void *state
)
{
}


static void
cupti_process_memcpy2
(
 CUpti_ActivityMemcpy2 *activity, 
 void *state
)
{
}


static void
cupti_process_memctr
(
 CUpti_ActivityUnifiedMemoryCounter *activity, 
 void *state
)
{
}


static void
cupti_process_activityAPI
(
 CUpti_ActivityAPI *activity,
 void *state
)
{
  // case CUPTI_ACTIVITY_KIND_DRIVER:
  // case CUPTI_ACTIVITY_KIND_KERNEL:
}


static void
cupti_process_runtime
(
 CUpti_ActivityEvent *activity, 
 void *state
)
{
}


static void
cupti_process_activity
(
 CUpti_Activity *activity,
 void *state
)
{
  switch (activity->kind) {

  case CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR:
    cupti_process_source_locator((CUpti_ActivitySourceLocator *) activity, 
				 state);
    break;

  case CUPTI_ACTIVITY_KIND_PC_SAMPLING:
    cupti_process_sample((CUpti_ActivityPCSampling2 *) activity, state);
    break;

  case CUPTI_ACTIVITY_KIND_PC_SAMPLING_RECORD_INFO:
    cupti_process_sampling_record_info
      ((CUpti_ActivityPCSamplingRecordInfo *) activity, state);
    break;

  case CUPTI_ACTIVITY_KIND_FUNCTION:
    cupti_process_function((CUpti_ActivityFunction *) activity, state);
    break;

  case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION: 
    cupti_process_correlation((CUpti_ActivityExternalCorrelation *) activity,
			      state);
    break;

  case CUPTI_ACTIVITY_KIND_MEMCPY: 
    cupti_process_memcpy((CUpti_ActivityMemcpy *) activity, state);
    break;

  case CUPTI_ACTIVITY_KIND_MEMCPY2: 
    cupti_process_memcpy2((CUpti_ActivityMemcpy2 *) activity, state);

  case CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER:
    cupti_process_memctr((CUpti_ActivityUnifiedMemoryCounter *) activity, state);
    break;

  case CUPTI_ACTIVITY_KIND_DRIVER:
  case CUPTI_ACTIVITY_KIND_KERNEL:
    cupti_process_activityAPI((CUpti_ActivityAPI *) activity, state);
    break;

  case CUPTI_ACTIVITY_KIND_RUNTIME:
    cupti_process_runtime((CUpti_ActivityEvent *) activity, state);
    break;

  default:
    cupti_process_unknown(activity, state);
    break;
  }
}
#endif 

