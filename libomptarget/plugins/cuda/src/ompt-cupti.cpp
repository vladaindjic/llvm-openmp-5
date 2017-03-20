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

#include <map>
#include <vector>

#include <dlfcn.h>

#include <ompt.h>



//******************************************************************************
// local includes 
//******************************************************************************

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
    dest->start_time = src->start_time;		\
    dest->end_time = src->end_time;		\
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
  macro(ompt_target_start_trace)		\
  macro(ompt_target_set_trace_native) 		\
  macro(ompt_target_get_time)			\
  macro(ompt_target_advance_buffer_cursor)	\
  macro(ompt_target_buffer_get_record_native)   \
  macro(ompt_target_buffer_get_record_type)	\
  macro(ompt_target_buffer_get_record_native_abstract)

#define FOREACH_MISSING_TARGET_FN(macro) 		\
  macro(ompt_target_start_trace)			\
  macro(ompt_target_set_trace_native)			\
  macro(ompt_target_get_time)				\
  macro(ompt_target_advance_buffer_cursor)		\
  macro(ompt_target_buffer_get_record_native)		\
  macro(ompt_target_buffer_get_record_type)		\
  macro(ompt_target_buffer_get_record_native_abstract)

//******************************************************************************
// types
//******************************************************************************

class ompt_device_info_t {
public:
  int device_id;
  int global_id;
  int id;
  ompt_get_target_info_inquiry_t get_target_info;
  ompt_target_buffer_request_callback_t request_callback;
  ompt_target_buffer_complete_callback_t complete_callback;
  ompt_device_info_t() : 
    device_id(0), global_id(0), id(0), get_target_info(0), 
    request_callback(0), complete_callback(0) 
  {
  };
}; 

typedef ompt_target_id_t (*ompt_target_operation_id_t)();

typedef CUptiResult (*cupti_enable_disable_fn)
(
 CUcontext context, 
 CUpti_ActivityKind kind
);

typedef std::map<int32_t, const char *> device_types_map_t; 

typedef void (*ompt_target_start_tool_t) (ompt_initialize_t);



//******************************************************************************
// constants
//******************************************************************************

static const char ompt_documentation[] =
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

static ompt_target_operation_id_t        ompt_target_operation_id;
static ompt_device_initialize_callback_t ompt_target_device_initialize;

static std::vector<ompt_device_info_t> device_info;

// temporary definitions of missing target entry points 
#define macro(fn)				\
  static ompt_interface_fn_t fn;

FOREACH_MISSING_TARGET_FN(macro); 
#undef macro

  
//----------------------------------------
// thread local data
//----------------------------------------

#if 0
thread_local ompt_native_abstract_t ompt_native_abstract;
#endif

thread_local ompt_target_id_t ompt_correlation_id;




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


static void
ompt_device_rtl_init
(
 ompt_function_lookup_t lookup, 
 const char *version_string,
 unsigned int version_number
)
{
  ompt_enabled = true;
  ompt_version_string = version_string;
  ompt_version_number = version_number;

  ompt_target_operation_id = 
    (ompt_target_operation_id_t) lookup("ompt_target_operation_id");

  ompt_target_device_initialize = 
    (ompt_device_initialize_callback_t) lookup("ompt_target_device_initialize");
}


void
ompt_device_rtl_init_device
(
 int omp_device_id
)
{
  int device_id = omp_device_id;

  if (ompt_target_device_initialize) {
    ompt_target_device_initialize
      (omp_device_id,
      ompt_device_get_type(device_id),
      (ompt_target_device_t) 0, // fixme
      ompt_device_lookup,
      ompt_documentation);
  }
}


static void 
ompt_init_device_infos
(
 int num_devices, 
 int first_global_id
)
{
  device_info.resize(num_devices);
  for (int i = 0; i < num_devices; i++) {
    device_info[i].local_id = i;
    device_info[i].global_id = first_global_id + i;
  }
}


void
ompt_init
(
 int num_devices
)
{
  static bool initialized;
  if (initialized == false) {
    ompt_target_start_tool_t libomptarget_start_tool_fn = 
      (ompt_target_start_tool_t) dlsym(NULL, "libomptarget_start_tool");
    if (libomptarget_start_tool_fn) {
      libomptarget_start_tool_fn(ompt_device_rtl_init);
    }
    ompt_init_device_infos(num_devices);
    initialized = true;
  }
}

#if 0

//----------------------------------------
// internal operations 
//----------------------------------------

#if 0
static inline int
ompt_get_device_id
(
 ompt_target_device_t *device
) 
{
  if (device) {
    int global_id = ((ompt_device_info_t *) device)->global_id;
    int device_id = global_id - device_offset;
    if (device_id >= 0 && device_id <= DeviceInfo.NumberOfDevices) 
      return device_id;
  }
  return NO_DEVICE;
}


//Generate Request Functions
#define MAKE_REQUEST_FN_NAME(i) cupti_buffer_request_ ## i
#define MAKE_REQUEST_FN(i) \
  static void								\
  MAKE_REQUEST_FN_NAME(i)						\
    (uint8_t **buffer,							\
     size_t *size,							\
     size_t *maxNumRecords)						\
  {									\
    *maxNumRecords = 0;							\
    DeviceInfo.ompt_info[i].request_callback(i, buffer, size);		\
  }

//Generate Complete Functions
#define MAKE_COMPLETE_FN_NAME(i) cupti_buffer_complete_ ## i

#define MAKE_COMPLETE_FN(i) \
  static void								\
  MAKE_COMPLETE_FN_NAME(i)						\
    (CUcontext ctx,							\
     uint32_t streamId,							\
     uint8_t *buffer,							\
     size_t size,							\
     size_t validSize)							\
  {									\
    ompt_target_buffer_cursor_t begin;					\
    DeviceInfo.ompt_info[i].complete_callback				\
      (i, (const ompt_target_buffer_t*)buffer, validSize, begin, true); \
  }

MAKE_REQUEST_FN(0)
MAKE_REQUEST_FN(1)
MAKE_REQUEST_FN(2)
MAKE_REQUEST_FN(3)
MAKE_REQUEST_FN(4)
MAKE_REQUEST_FN(5)
MAKE_REQUEST_FN(6)
MAKE_REQUEST_FN(7)
MAKE_REQUEST_FN(8)
MAKE_REQUEST_FN(9)

MAKE_COMPLETE_FN(0)
MAKE_COMPLETE_FN(1)
MAKE_COMPLETE_FN(2)
MAKE_COMPLETE_FN(3)
MAKE_COMPLETE_FN(4)
MAKE_COMPLETE_FN(5)
MAKE_COMPLETE_FN(6)
MAKE_COMPLETE_FN(7)
MAKE_COMPLETE_FN(8)
MAKE_COMPLETE_FN(9)

CUpti_BuffersCallbackRequestFunc request_fns[] = {
MAKE_REQUEST_FN_NAME(0),
MAKE_REQUEST_FN_NAME(1),
MAKE_REQUEST_FN_NAME(2),
MAKE_REQUEST_FN_NAME(3),
MAKE_REQUEST_FN_NAME(4),
MAKE_REQUEST_FN_NAME(5),
MAKE_REQUEST_FN_NAME(6),
MAKE_REQUEST_FN_NAME(7),
MAKE_REQUEST_FN_NAME(8),
MAKE_REQUEST_FN_NAME(9)
};

CUpti_BuffersCallbackCompleteFunc complete_fns[] = {
MAKE_COMPLETE_FN_NAME(0),
MAKE_COMPLETE_FN_NAME(1),
MAKE_COMPLETE_FN_NAME(2),
MAKE_COMPLETE_FN_NAME(3),
MAKE_COMPLETE_FN_NAME(4),
MAKE_COMPLETE_FN_NAME(5),
MAKE_COMPLETE_FN_NAME(6),
MAKE_COMPLETE_FN_NAME(7),
MAKE_COMPLETE_FN_NAME(8),
MAKE_COMPLETE_FN_NAME(9),
};
#endif

//----------------------------------------
// OMPT buffer management support
//----------------------------------------

static ompt_native_abstract_t *
ompt_abstract_init()
{
  ompt_native_abstract_t *a = &ompt_native_abstract;
  a->rclass = ompt_record_native_class_event;
  a->hwid = ompt_hwid_none;
  a->start_time = ompt_time_none;
  a->end_time = ompt_time_none;
  return a;
}


//----------------------------------------
// OMPT buffer management interface
//----------------------------------------


static int32_t
ompt_target_advance_buffer_cursor
(
 ompt_target_buffer_t *buffer,
 size_t validSize,
 ompt_target_buffer_cursor_t current,
 ompt_target_buffer_cursor_t *next
)
{
  DECLARE_CAST(CUpti_Activity, cursor, current);
  bool result = cupti_buffer_cursor_advance(buffer, validSize, &cursor);
  if (result) {
    *next = (ompt_target_buffer_cursor_t) cursor;
  }
  return result;
}


static ompt_record_type_t 
ompt_get_record_type(
  ompt_target_buffer_t *buffer,
  size_t validSize,
  ompt_target_buffer_cursor_t current
)
{
  DECLARE_CAST(CUpti_Activity, activity, current);
  return (cupti_buffer_cursor_isvalid(buffer, validSize, activity) ?
	  ompt_record_native : ompt_record_invalid);
}


static int32_t
ompt_target_advance_buffer_cursor
(
 ompt_target_buffer_t *buffer,
 size_t validSize,
 ompt_target_buffer_cursor_t current,
 ompt_target_buffer_cursor_t *next
)
{
  DECLARE_CAST(CUpti_Activity, cursor, current);
  CUptiResult result = cuptiActivityGetNextRecord(buffer, validSize, &cursor);
  *next = (ompt_target_buffer_cursor_t) cursor;
  return (result != CUPTI_SUCCESS);
}


static ompt_record_type_t 
ompt_get_record_type(
  ompt_target_buffer_t *buffer,
  size_t validSize,
  ompt_target_buffer_cursor_t current
)
{
  DECLARE_CAST(CUpti_Activity, activity, current);
  CUptiResult status = cuptiActivityGetNextRecord(buffer, validSize, &activity);
  return (status == CUPTI_SUCCESS) ? ompt_record_native : ompt_record_invalid;
}


static void*
ompt_get_record_native
(
 ompt_target_buffer_t *buffer,
 ompt_target_buffer_cursor_t current,
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


static ompt_record_native_abstract_t * 
ompt_get_record_abstract
(
  CUpti_Activity *activity
)
{
  ompt_record_native_abstract_t *a = ompt_abstract_init();

  switch (activity->kind) {

  case CUPTI_ACTIVITY_KIND_MEMCPY: {
    DECLARE_CAST(CUpti_ActivityMemcpy, rhs, cursor);
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

  }
  return abs;
}


//----------------------------------------
// OMPT device tracing control
//----------------------------------------


static int32_t
ompt_set_trace_native
(
 ompt_target_device_t *device, 
 bool enable, 
 uint32_t flags 
) 
{ 
  int result = ;
  int device_id = ompt_get_device_id(device);
  if (device_id != NO_DEVICE) {
    int result = 0;
    int status;

    const char *action_string = enable ? "enable" : "disable";

#define set_trace(flag, activity)					\
    if (flags & flag) {							\
      int action_result = cupti_set_monitoring(activities, enable);	\
			   OMPT_TRACING_OK :				\
			   OMPT_TRACING_FAILED);			\
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
	      OMPT_TRACING_SOME :
	      OMPT_TRACING_ALL);
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
 ompt_target_device_t *device,
 int begin_pause
)
{
  bool result = false;
  int device_id = ompt_get_device_id(device);

  CUresult cu_result = cuCtxSetCurrent(DeviceInfo.Contexts[device_id]);
  if (cu_result == CUDA_SUCCESS) {
    if (begin_pause) {
    } else {
      CUptiResult cupti_result =
	cuptiActivityRegisterCallbacks(request_fns[device_id],
				       complete_fns[device_id]);

      result = (cupti_result == CUPTI_SUCCESS);
    }
  }

  return result;
}


static bool
ompt_start_trace
(
 ompt_target_device_t *device,
 ompt_target_buffer_request_callback_t request,
 ompt_target_buffer_complete_callback_t complete
)
{
  DeviceInfo.ompt_info[device_id].request_callback = request;
  DeviceInfo.ompt_info[device_id].complete_callback = complete;

  return ompt_pause_trace(0);
}


static bool
ompt_stop_trace
(
 ompt_target_device_t *device,
)
{
  ompt_pause_trace(1);
  // FIXME: flush
}


//*************************************************************************

//----------------------------------------
// internal functions
//----------------------------------------
  

//----------------------------------------
// OMPT device time
//----------------------------------------


static bool
ompt_get_device_time
(
 ompt_target_device_t *device, 
 ompt_target_time_t *time
)
{
  int device_id = ompt_get_device_id(device);
  CUcontext context = DeviceInfo.Contexts[device_id];

  return cupti_device_get_timestamp(context, (uint64_t *) time);
}


static double
ompt_translate_time
(
 ompt_target_device_t *device, 
 ompt_target_time_t time
)
{
  double omp_time;

  assert(0);
  // record OpenMP time when initialized

  return omp_time;
}

#endif
