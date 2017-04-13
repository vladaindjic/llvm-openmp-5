//===----RTLs/cuda/src/rtl.cpp - Target RTLs Implementation ------- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//
//
// OMPT for NVIDIA GPU 
//
//===----------------------------------------------------------------------===//



//******************************************************************************
// global includes 
//******************************************************************************

#include <cassert>
#include <map>
#include <vector>

#include <dlfcn.h>

#include <ompt.h>




//******************************************************************************
// local includes 
//******************************************************************************

#include "omptarget.h"

#include "rtl.h"

#include "cupti.hpp"
#include "cuda.hpp"



//******************************************************************************
// macros
//******************************************************************************

#define NO_DEVICE -1
#define DEVICE_TYPE_NCHARS 1024

#define DECLARE_CAST(t, x, y) t *x = (t *) y

#define COPY_TIMES(dest, src)			\
  {						\
    dest->start_time = src->start;		\
    dest->end_time = src->end;			\
  }


#define OMPT_TRACING_OK      4
#define OMPT_TRACING_FAILED  2

#define OMPT_TRACING_ALL   4
#define OMPT_TRACING_SOME  3
#define OMPT_TRACING_NONE  1
#define OMPT_TRACING_ERROR 0

#define FOREACH_FLAGS(macro)						\
  macro(ompt_native_data_motion_explicit, data_motion_explicit_activities) \
  macro(ompt_native_data_motion_implicit, data_motion_implicit_activities) \
  macro(ompt_native_kernel_invocation, kernel_invocation_activities)	\
  macro(ompt_native_kernel_execution, kernel_execution_activities)	\
  macro(ompt_native_driver, driver_activities)				\
  macro(ompt_native_runtime, runtime_activities)			\
  macro(ompt_native_overhead, overhead_activities)

#define FOREACH_TARGET_FN(macro) 		\
  macro(ompt_get_device_time)			\
  macro(ompt_translate_time)			\
  macro(ompt_set_trace_native)			\
  macro(ompt_start_trace)			\
  macro(ompt_pause_trace)			\
  macro(ompt_stop_trace)			\
  macro(ompt_advance_buffer_cursor)		\
  macro(ompt_get_record_type)			\
  macro(ompt_get_record_native)			\
  macro(ompt_get_record_abstract)


//******************************************************************************
// types
//******************************************************************************

class ompt_device_info_t {
public:
  int initialized;
  int relative_id;
  int global_id;
  CUcontext context;
  ompt_callback_buffer_request_t request_callback;
  ompt_callback_buffer_complete_t complete_callback;
  ompt_device_info_t() : 
    initialized(0), relative_id(0), global_id(0), context(0), 
    request_callback(0), complete_callback(0)
  {
  };
}; 

typedef ompt_target_id_t (*libomptarget_get_target_info_t)();

typedef CUptiResult (*cupti_enable_disable_fn)
(
 CUcontext context, 
 CUpti_ActivityKind kind
);

typedef std::map<int32_t, const char *> device_types_map_t; 

typedef std::map<CUcontext, int> context_to_device_map_t; 

typedef void (*ompt_target_start_tool_t) (ompt_initialize_t);



//******************************************************************************
// constants
//******************************************************************************

static const char *ompt_documentation =
  #include "ompt-documentation.h"
  ;



//******************************************************************************
// global data
//******************************************************************************

//--------------------------
// values set by initializer
//--------------------------

static bool ompt_enabled = false;

static const char *ompt_version_string = 0;
static int ompt_version_number = 0;

static libomptarget_get_target_info_t        libomptarget_get_target_info;
static ompt_callback_device_initialize_t libomp_callback_device_initialize;

static std::vector<ompt_device_info_t> device_info;

static context_to_device_map_t context_to_device_map;

  

//----------------------------------------
// thread local data
//----------------------------------------

thread_local ompt_record_abstract_t ompt_record_abstract;
thread_local ompt_target_id_t ompt_correlation_id;



//******************************************************************************
// private operations
//******************************************************************************


// device needs
//   local device id
//   OpenMP device id
//   device name
//----------------------------------------
// OMPT initialization
//----------------------------------------


static const char *
ompt_device_get_type
(
 int32_t device_id
)
{
  static device_types_map_t device_types_map;
  device_types_map_t::iterator it = device_types_map.find(device_id);
  
  const char *name;
  if (it != device_types_map.end()) {
    name = it->second;
  } else {
    int major, minor;
    char device_type[DEVICE_TYPE_NCHARS];

    const char *device_name = cuda_device_get_name(device_id);
    cuda_compute_capability(device_id, &major, &minor);

    sprintf(device_type, 
	    "NVIDIA; %s; Compute Capability %d.%d", 
	    device_name, major, minor);

    name = strdup(device_type);

    std::pair<int32_t, const char *> value(device_id, name);
    device_types_map.insert(value);
  }
  return name;
}



static void
ompt_device_rtl_init
(
 ompt_function_lookup_t lookup, 
 const char *version_string,
 unsigned int version_number
)
{
  DP("enter ompt_device_rtl_init\n");

  ompt_enabled = true;
  ompt_version_string = version_string;
  ompt_version_number = version_number;

  libomptarget_get_target_info = 
    (libomptarget_get_target_info_t) lookup("libomptarget_get_target_info");

  DP("libomptarget_get_target_info = %p\n", (void *) (uint64_t) libomptarget_get_target_info);

  libomp_callback_device_initialize = 
    (ompt_callback_device_initialize_t) lookup("libomp_callback_device_initialize");

  DP("libomp_callback_device_initialize = %p\n",  (void *) (uint64_t) libomp_callback_device_initialize);
  
  DP("exit ompt_device_rtl_init\n");
}


static void 
ompt_device_infos_alloc
(
 int num_devices 
)
{
  device_info.resize(num_devices);
}


static bool 
ompt_device_info_init
(
 int device_id, 
 int omp_device_id,
 CUcontext context
)
{
  bool valid_device = device_id < (int) device_info.size();

  if (valid_device) {
    device_info[device_id].relative_id = device_id;
    device_info[device_id].global_id = omp_device_id;
    device_info[device_id].context = context;
    device_info[device_id].initialized = 1;
  }

  return valid_device;
}

//----------------------------------------
// internal operations 
//----------------------------------------

static inline int
ompt_get_device_id
(
 ompt_device_t *device
) 
{
  ompt_device_info_t *dptr = (ompt_device_info_t*) device;
  return dptr ? dptr->relative_id : NO_DEVICE;
}


//----------------------------------------
// OMPT buffer management support
//----------------------------------------

static ompt_record_abstract_t *
ompt_abstract_init()
{
  ompt_record_abstract_t *a = &ompt_record_abstract;
  a->rclass = ompt_record_native_event;
  a->hwid = ompt_hwid_none;
  a->start_time = ompt_device_time_none;
  a->end_time = ompt_device_time_none;
  return a;
}


//----------------------------------------
// OMPT buffer management interface
//----------------------------------------


static int32_t
ompt_advance_buffer_cursor
(
 ompt_buffer_t *buffer,
 size_t size,
 ompt_buffer_cursor_t current,
 ompt_buffer_cursor_t *next
)
{
  DECLARE_CAST(CUpti_Activity, cursor, current);
  bool result = cupti_buffer_cursor_advance(buffer, size, &cursor);
  if (result) {
    *next = (ompt_buffer_cursor_t) cursor;
  }
  return result;
}


static ompt_record_type_t 
ompt_get_record_type(
  ompt_buffer_t *buffer,
  size_t validSize,
  ompt_buffer_cursor_t current
)
{
  DECLARE_CAST(CUpti_Activity, activity, current);
  return (cupti_buffer_cursor_isvalid(buffer, validSize, activity) ?
	  ompt_record_native : ompt_record_invalid);
}


static void*
ompt_get_record_native
(
 ompt_buffer_t *buffer,
 ompt_buffer_cursor_t current,
 ompt_target_id_t *host_opid
)
{
  DECLARE_CAST(CUpti_Activity, activity, current);
  if (activity->kind == CUPTI_ACTIVITY_KIND_CONTEXT){
    DECLARE_CAST(CUpti_ActivityContext, context, current);
    ompt_correlation_id = context->contextId;
  }
  *host_opid = ompt_correlation_id;
  return activity;
}


static ompt_record_abstract_t * 
ompt_get_record_abstract
(
  CUpti_Activity *activity
)
{
  ompt_record_abstract_t *abs = ompt_abstract_init();

  switch (activity->kind) {

  case CUPTI_ACTIVITY_KIND_MEMCPY: {
    DECLARE_CAST(CUpti_ActivityMemcpy, rhs, activity);
    abs->type = "MEMCPY EXPLICIT";
    COPY_TIMES(abs, rhs);
    break;
  }

  case CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER: {
    DECLARE_CAST(CUpti_ActivityUnifiedMemoryCounter2, rhs, activity);
    abs->type = "MEMCPY IMPLICIT";
    COPY_TIMES(abs, rhs);
    break;
  }

  case CUPTI_ACTIVITY_KIND_KERNEL:
  case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
    DECLARE_CAST(CUpti_ActivityKernel3, rhs, activity);
    abs->type = "KERNEL INVOCATION";
    COPY_TIMES(abs, rhs);
    break;
  }

  case CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR: {
    abs->type = "KERNEL SOURCE LOCATOR";
    break;
  }

  case CUPTI_ACTIVITY_KIND_PC_SAMPLING: {
    abs->type = "KERNEL PC SAMPLE";
    break;
  }

  case CUPTI_ACTIVITY_KIND_PC_SAMPLING_RECORD_INFO: {
    abs->type = "KERNEL PC SAMPLING INFO";
    break;
  }

  case CUPTI_ACTIVITY_KIND_FUNCTION: {
    abs->type = "FUNCTION";
    break;
  }		

  case CUPTI_ACTIVITY_KIND_DRIVER: {
    DECLARE_CAST(CUpti_ActivityAPI, rhs, activity);
    abs->type = "DRIVER";
    COPY_TIMES(abs, rhs);
    abs->hwid = rhs->threadId;
    break;
  }

  case CUPTI_ACTIVITY_KIND_RUNTIME: {
    DECLARE_CAST(CUpti_ActivityAPI, rhs, activity);
    abs->type = "RUNTIME";
    COPY_TIMES(abs, rhs);
    abs->hwid = rhs->threadId;
    break;
  }

  case CUPTI_ACTIVITY_KIND_OVERHEAD: {
    DECLARE_CAST(CUpti_ActivityOverhead, rhs, activity);
    abs->type = "OVERHEAD";
    COPY_TIMES(abs, rhs);
    break;
  }

  default:
    DP("CUPTI activity kind %d not handled by ompt-cupti.cpp\n", activity->kind);
    break;
  }
  return abs;
}


void
device_completion_callback
(
 uint64_t relative_device_id,
 CUpti_Activity *start,
 CUpti_Activity *end
)
{
  const int BUFFER_NOT_OWNED = 0;
  uint8_t *ustart = (uint8_t *) start;
  uint8_t *uend = (uint8_t *) end;
  size_t bytes = uend - ustart;
  if (bytes != 0) {
    device_info[relative_device_id].complete_callback
      (device_info[relative_device_id].global_id, (ompt_buffer_t *) ustart, bytes, 
       (ompt_buffer_cursor_t *) ustart, BUFFER_NOT_OWNED);
  }
}


static void 
cupti_buffer_completion_callback
(
 CUcontext context,  // NULL since CUDA 6.0
 uint32_t streamId, // unused?
 uint8_t*  buffer,  
 size_t size,  
 size_t validSize 
)
{
  CUpti_Activity *activity = NULL; // signal advance to return pointer to first record

  // set activity to point to first record
  bool status = cupti_buffer_cursor_advance(buffer, validSize, &activity);

  if (status) { // so far, so good ...
    uint64_t relative_device_id = 0;
    CUpti_Activity *start = activity;
    while (status) {
      status = cupti_buffer_cursor_advance(buffer, validSize, &activity);
      
      if (activity->kind == CUPTI_ACTIVITY_KIND_CONTEXT) {
	device_completion_callback(relative_device_id, start, activity);
	start = activity;
	DECLARE_CAST(CUpti_ActivityContext, ac, activity);
	relative_device_id = ac->deviceId;
      }
    }
    device_completion_callback(relative_device_id, start, activity);
  }
  free(buffer);
}



//****************************************************************************
// TARGET CONTROL API
//****************************************************************************

//----------------------------------------
// OMPT device tracing control
//----------------------------------------


static int
ompt_set_trace_native
(
 ompt_device_t *device, 
 int enable, 
 int flags 
) 
{ 
  int relative_device_id = ompt_get_device_id(device);
  if (relative_device_id != NO_DEVICE) {
    int result = 0;

#define set_trace(flag, activities)					\
    if (flags & flag) {							\
      int action_result = cupti_set_monitoring(activities, enable);	\
      result |= action_result;						\
      flags ^= flag;							\
    } 
	
    FOREACH_FLAGS(set_trace);
	
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION);

    if (flags) { 
      return OMPT_TRACING_ERROR; // unhandled flags
    }

    if (result & OMPT_TRACING_OK) {
      return ((result & OMPT_TRACING_FAILED) ? 
	      OMPT_TRACING_SOME : OMPT_TRACING_ALL);
    }

    if (result & OMPT_TRACING_FAILED) {
      return OMPT_TRACING_NONE;
    }
	
  }
  return OMPT_TRACING_ERROR;
}


static bool
ompt_pause_trace
(
 ompt_device_t *device,
 int begin_pause
)
{
  bool result = false;
  int device_id = ompt_get_device_id(device);

  bool set_result = cuda_context_set(device_info[device_id].context);
  if (set_result == CUDA_SUCCESS) {
    if (begin_pause) {
      cupti_stop();
    } else {
      CUptiResult cupti_result =
	cuptiActivityRegisterCallbacks(cupti_buffer_alloc,
				       cupti_buffer_completion_callback);

      result = (cupti_result == CUPTI_SUCCESS);
    }
  }

  return result;
}


static bool
ompt_start_trace
(
 ompt_device_t *device,
 ompt_callback_buffer_request_t request,
 ompt_callback_buffer_complete_t complete
)
{
  int device_id = ompt_get_device_id(device);
  device_info[device_id].request_callback = request;
  device_info[device_id].complete_callback = complete;

  return ompt_pause_trace(device, 0);
}


static bool
ompt_stop_trace
(
 ompt_device_t *device
)
{
  return ompt_pause_trace(device, 1);
}



//*************************************************************************

//----------------------------------------
// internal functions
//----------------------------------------
  

//----------------------------------------
// OMPT device time
//----------------------------------------


#define ompt_device(d) ((ompt_device_info_t *) d)

static ompt_device_time_t 
ompt_get_device_time
(
 ompt_device_t *device
)
{
  CUcontext context = ompt_device(device)->context;
  uint64_t time;
  cupti_device_get_timestamp(context, &time);
  return (ompt_device_time_t) time; 
}


static double
ompt_translate_time
(
 ompt_device_t *device, 
 ompt_device_time_t time
)
{
  double omp_time;

  assert(0);
  // record OpenMP time when initialized

  return omp_time;
}


//******************************************************************************
// interface operations
//******************************************************************************

extern "C" {

__attribute__ (( weak ))
void
libomptarget_start_tool
(
 ompt_initialize_t ompt_init
)
{
  // no initialization of OMPT for device-specific rtl unless 
  // libomptarget implements this function
}

}


void
ompt_init
(
 int num_devices
)
{
  static bool initialized;

  DP("enter ompt_init\n");

  if (initialized == false) {
    void *vptr = dlsym(NULL, "libomptarget_start_tool");
    ompt_target_start_tool_t libomptarget_start_tool_fn = 
      reinterpret_cast<ompt_target_start_tool_t>(reinterpret_cast<long>(vptr));

    if (libomptarget_start_tool_fn) {
      libomptarget_start_tool_fn(ompt_device_rtl_init);
    }
    ompt_device_infos_alloc(num_devices);
    initialized = true;
  }

  DP("exit ompt_init\n");
}


static ompt_interface_fn_t 
ompt_device_lookup
(
 const char *s
)
{
#define macro(fn) \
  if (strcmp(s, #fn) == 0) return (ompt_interface_fn_t) fn;

  FOREACH_TARGET_FN(macro);

#undef macro

  return (ompt_interface_fn_t) 0;
}


void
ompt_device_init
(
 int device_id, 
 int omp_device_id,
 CUcontext context 
)
{
  DP("enter ompt_device_init\n");

  DP("libomp_callback_device_initialize = %p\n", (void *) (uint64_t) libomp_callback_device_initialize);
  if (libomp_callback_device_initialize) {
    if (ompt_device_info_init(device_id, omp_device_id, context)) {

      DP("calling libomp_callback_device_initialize\n");

      libomp_callback_device_initialize
        (omp_device_id,
         ompt_device_get_type(omp_device_id),
         (ompt_device_t *) &device_info[device_id], 
         ompt_device_lookup,
         ompt_documentation);
    }
  }

  DP("exit ompt_device_init\n");
}
