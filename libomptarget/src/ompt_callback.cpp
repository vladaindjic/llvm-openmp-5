//===------ omptarget.cpp - Target independent OpenMP target RTL -- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of ompt callback interfaces
//
//===----------------------------------------------------------------------===//

#include "ompt_callback.h"

#include <atomic>
#include <cstring>
#include <dlfcn.h>
#include <assert.h>

#include <ompt.h>
#include "private.h"

/*******************************************************************************
 * macros
 *******************************************************************************/

#define OMPT_CALLBACK(fn, args) if (ompt_enabled && fn) fn args
#define fnptr_to_ptr(fn) (void *) fn

/*******************************************************************************
 * class
 *******************************************************************************/

#if 0
class libomptarget_rtl_finalizer_t : std::list<ompt_finalize_t> {
public:
  void register_rtl(ompt_finalize_t fn) {
    push_back(fn);
  };

  void finalize() {
    for(ompt_finalize_t fn : *this) {
      fn(NULL);
    }
  };
};
#else

class libomptarget_rtl_finalizer_t : std::list<ompt_finalize_t> {
public:
  libomptarget_rtl_finalizer_t() : fn(0) {};
  void register_rtl(ompt_finalize_t _fn) {
    assert(fn == 0);
    fn = _fn;
  };

  void finalize() {
    if (fn) fn(NULL);
  };
  ompt_finalize_t fn;
};
#endif

/*****************************************************************************
 * global data
 *****************************************************************************/

static bool ompt_enabled = false;

typedef void (*ompt_set_frame_reenter_t)(void *);  
typedef ompt_data_t *(*ompt_get_task_data_t)();  

static ompt_set_frame_reenter_t ompt_set_frame_reenter_fn = 0;
static ompt_get_task_data_t ompt_get_task_data_fn = 0;

#define declare_name(name)			\
  static name ## _t name ## _fn = 0; 

FOREACH_OMPT_TARGET_CALLBACK(declare_name)

#undef declare_name

static const char *libomp_version_string;
static unsigned int libomp_version_number;

static libomptarget_rtl_finalizer_t libomptarget_rtl_finalizer;

/*****************************************************************************
 * Thread local data
 *****************************************************************************/

static thread_local uint64_t ompt_target_region_id = 1;
static thread_local uint64_t ompt_target_region_opid = 1;

static std::atomic<uint64_t> ompt_target_region_id_ticket(1);
static std::atomic<uint64_t> ompt_target_region_opid_ticket(1);

/*****************************************************************************
 * OMPT callbacks
 *****************************************************************************/

uint64_t OmptCallback::target_region_begin() {
  uint64_t retval = 0;
  if (ompt_enabled) {
    ompt_target_region_id = ompt_target_region_id_ticket.fetch_add(1);
    retval = ompt_target_region_id;
    DP("in OmptCallback::target_region_begin (retval = %lu)\n", retval);
  } 
  return retval;
}

uint64_t OmptCallback::target_region_end() {
  uint64_t retval = 0;
  if (ompt_enabled) {
    retval = ompt_target_region_id;
    ompt_target_region_id = 0;
    DP("in OmptCallback::target_region_end (retval = %lu)\n", retval);
  } 
  return retval;
}

void OmptCallback::target_operation_begin() {
  if (ompt_enabled) {
    ompt_target_region_opid = ompt_target_region_opid_ticket.fetch_add(1);
    if (ompt_set_frame_reenter_fn) {
      ompt_set_frame_reenter_fn(_codeptr);
    }
    DP("in ompt_target_region_begin (ompt_target_region_opid = %lu)\n", 
       ompt_target_region_opid);
  }
}

void OmptCallback::target_operation_end() {
  if (ompt_enabled) {
    if (ompt_set_frame_reenter_fn) {
      ompt_set_frame_reenter_fn(0);
    }
    DP("in ompt_target_region_end (ompt_target_region_opid = %lu)\n", 
       ompt_target_region_opid);
  }
}

void OmptCallback::target_data_alloc(int64_t device_id, void *tgt_ptr_begin, size_t size) {
  OMPT_CALLBACK(ompt_callback_target_data_op_fn, 
    (ompt_target_region_id, ompt_target_region_opid, ompt_target_data_alloc,
     tgt_ptr_begin, device_id, NULL, 0, size, _codeptr));
}

void OmptCallback::target_data_submit(int64_t device_id, void *tgt_ptr_begin,
  void *hst_ptr_begin, size_t size) {
  OMPT_CALLBACK(ompt_callback_target_data_op_fn, 
    (ompt_target_region_id, ompt_target_region_opid, ompt_target_data_transfer_to_device,
     hst_ptr_begin, 0, tgt_ptr_begin, device_id, size, _codeptr));
}

void OmptCallback::target_data_delete(int64_t device_id, void *tgt_ptr_begin) {
  OMPT_CALLBACK(ompt_callback_target_data_op_fn, 
    (ompt_target_region_id, ompt_target_region_opid, ompt_target_data_delete,
     tgt_ptr_begin, device_id, NULL, 0, 0, _codeptr));
}

void OmptCallback::target_data_retrieve(int64_t device_id, void *hst_ptr_begin,
  void *tgt_ptr_begin, size_t size) {
  OMPT_CALLBACK(ompt_callback_target_data_op_fn,
    (ompt_target_region_id, ompt_target_region_opid, ompt_target_data_transfer_from_device,
     tgt_ptr_begin, device_id, hst_ptr_begin, 0, size, _codeptr));
} 

void OmptCallback::target_submit() {
  OMPT_CALLBACK(ompt_callback_target_submit_fn, 
		(ompt_target_region_id, ompt_target_region_opid, 0));
}

void OmptCallback::target_enter_data(int64_t device_id) {
  OMPT_CALLBACK(ompt_callback_target_fn, 
    (ompt_target_enter_data, 
     ompt_scope_begin,
     device_id,
     ompt_get_task_data_fn(), 
     ompt_target_region_id, 
     _codeptr
    )); 
}

void OmptCallback::target_exit_data(int64_t device_id) {
  OMPT_CALLBACK(ompt_callback_target_fn, 
    (ompt_target_exit_data, 
     ompt_scope_begin,
     device_id,
     ompt_get_task_data_fn(), 
     ompt_target_region_id, 
     _codeptr
    )); 
}

void OmptCallback::target_update(int64_t device_id) {
  OMPT_CALLBACK(ompt_callback_target_fn, 
    (ompt_target_update, 
     ompt_scope_begin,
     device_id,
     ompt_get_task_data_fn(), 
     ompt_target_region_id, 
     _codeptr
    )); 
}

void OmptCallback::target(int64_t device_id) {
  OMPT_CALLBACK(ompt_callback_target_fn, 
	  (ompt_target, 
	   ompt_scope_begin,
	   device_id,
	   ompt_get_task_data_fn(), 
	   ompt_target_region_id, 
     _codeptr
	   )); 
}

/*****************************************************************************
 * OMPT interface operations
 *****************************************************************************/

static int libomptarget_ompt_initialize(ompt_function_lookup_t lookup,
  int initial_device_num, ompt_data_t *tool_data) {
  DP("enter libomptarget_ompt_initialize!\n");

  ompt_enabled = true;

#define ompt_bind_name(fn) \
  fn ## _fn = (fn ## _t ) lookup(#fn); DP("%s=%p\n", #fn, fnptr_to_ptr(fn ## _fn));

  ompt_bind_name(ompt_set_frame_reenter);	
  ompt_bind_name(ompt_get_task_data);	

#undef ompt_bind_name

#define ompt_bind_callback(fn) \
  fn ## _fn = (fn ## _t ) lookup(#fn); \
  DP("%s=%p\n", #fn, fnptr_to_ptr(fn ## _fn));

  FOREACH_OMPT_TARGET_CALLBACK(ompt_bind_callback)

#undef ompt_bind_callback

  DP("exit libomptarget_ompt_initialize!\n");

  return 0;
}

static void libomptarget_ompt_finalize(ompt_data_t *data) {
  DP("enter libomptarget_ompt_finalize!\n");

  libomptarget_rtl_finalizer.finalize(); 

  ompt_enabled = false;

  DP("exit libomptarget_ompt_finalize!\n");
}

static ompt_interface_fn_t libomptarget_rtl_fn_lookup(const char *fname) {
#define lookup_libomp_fn(fn) \
  if (strcmp(fname, #fn) == 0) return (ompt_interface_fn_t) fn ## _fn;

  FOREACH_OMPT_TARGET_CALLBACK(lookup_libomp_fn)

#undef lookup_libomp_fn

  return 0;
}

typedef void (*libomp_libomptarget_ompt_init_t) (ompt_start_tool_result_t*);

__attribute__ (( weak ))
void libomp_libomptarget_ompt_init(ompt_start_tool_result_t *result) {
  // no initialization of OMPT for libomptarget unless 
  // libomp implements this function
  DP("in dummy libomp_libomptarget_ompt_init\n");
}

void ompt_init() {
  static ompt_start_tool_result_t libomptarget_ompt_result;
  static bool initialized = false;

  if (initialized == false) {
    libomptarget_ompt_result.initialize = libomptarget_ompt_initialize;
    libomptarget_ompt_result.finalize   = libomptarget_ompt_finalize;
    
    DP("in ompt_init\n");
    libomp_libomptarget_ompt_init_t libomp_libomptarget_ompt_init_fn = 
      (libomp_libomptarget_ompt_init_t) (uint64_t) dlsym(NULL, "libomp_libomptarget_ompt_init");

    if (libomp_libomptarget_ompt_init_fn) {
      libomp_libomptarget_ompt_init_fn(&libomptarget_ompt_result);
    }
    initialized = true;
  }
}

extern "C" {

void libomptarget_rtl_ompt_init(ompt_start_tool_result_t *result) {
  DP("enter libomptarget_rtl_ompt_init\n");
  if (ompt_enabled && result) {
    libomptarget_rtl_finalizer.register_rtl(result->finalize); 
    result->initialize(libomptarget_rtl_fn_lookup, 0, NULL);
  }
  DP("leave libomptarget_rtl_ompt_init\n");
}

}
