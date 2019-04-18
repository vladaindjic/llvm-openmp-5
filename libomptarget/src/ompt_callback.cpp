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

#include <ompt.h>

#include "private.h"

/*******************************************************************************
 * macros
 *******************************************************************************/

#define OMPT_CALLBACK(fn, args) if (ompt_enabled && fn) fn args

/*******************************************************************************
 * class
 *******************************************************************************/

class libomptarget_rtl_fns_t : std::list<ompt_fns_t *> {
public:
  void register_rtl(ompt_fns_t *fns) {
    push_back(fns);
  };
  void finalize() {
    for(ompt_fns_t *fns : *this) {
      fns->finalize(fns);
    }
  };
};

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

static libomptarget_rtl_fns_t libomptarget_rtl_fns;

typedef void (*libomp_libomptarget_ompt_init_t) (ompt_fns_t*);

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
    _ompt_target_region_id = ompt_target_region_id_ticket.fetch_add(1);
    retval = _ompt_target_region_id;
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

void OmptCallback::target_data_alloc(void *TgtPtrBegin, size_t size) {
  OMPT_CALLBACK(ompt_callback_target_data_op_fn, 
     (ompt_target_region_id, ompt_target_region_opid, 
      ompt_target_data_alloc, 0, TgtPtrBegin, Size));
}

void OmptCallback::target_data_submit(void *HstPtrBegin, void *TgtPtrBegin, size_t size) {
  OMPT_CALLBACK(ompt_callback_target_data_op_fn, 
     (ompt_target_region_id, ompt_target_region_opid, 
      ompt_target_data_transfer_to_dev, HstPtrBegin, TgtPtrBegin, Size));
}

void OmptCallback::target_data_delete(void *TgtPtrBegin) {
  OMPT_CALLBACK(ompt_callback_target_data_op_fn, 
     (ompt_target_region_id, ompt_target_region_opid, 
      ompt_target_data_delete, 0, TgtPtrBegin, 0));
}

void OmptCallback::target_data_submit(void *HstPtrBegin, void *TgtPtrBegin, size_t size) {
  OMPT_CALLBACK(ompt_callback_target_data_op_fn, 
     (ompt_target_region_id, ompt_target_region_opid, 
      ompt_target_data_transfer_from_dev, HstPtrBegin, TgtPtrBegin, Size));
} 

void OmptCallback::target_submit() {
  OMPT_CALLBACK(ompt_callback_target_submit_fn, 
		(ompt_target_region_id, ompt_target_region_opid));
}

void OmptCallback::target_enter_data(int64_t device_id) {
  OMPT_CALLBACK(ompt_callback_target_fn, 
    (ompt_task_target_enter_data, 
     ompt_scope_begin,
     device_id,
     ompt_get_task_data_fn(), 
     ompt_target_region_id, 
     _codeptr
    )); 
}

void OmptCallback::target_exit_data(int64_t device_id) {
  OMPT_CALLBACK(ompt_callback_target_fn, 
    (ompt_task_target_exit_data, 
     ompt_scope_begin,
     device_id,
     ompt_get_task_data_fn(), 
     ompt_target_region_id, 
     _code_ptr
    )); 
}

void OmptCallback::target_update(int64_t device_id) {
  OMPT_CALLBACK(ompt_callback_target_fn, 
    (ompt_task_target_update, 
     ompt_scope_begin,
     device_id,
     ompt_get_task_data_fn(), 
     ompt_target_region_id, 
     _code_ptr
    )); 
}

void OmptCallback::target() {
  OMPT_CALLBACK(ompt_callback_target_fn, 
	  (ompt_task_target, 
	   ompt_scope_begin,
	   device_id,
	   ompt_get_task_data_fn(), 
	   ompt_target_region_id, 
     _code_ptr
	   )); 
}

/*****************************************************************************
 * OMPT interface operations
 *****************************************************************************/

__attribute__ (( weak ))
static void libomp_libomptarget_ompt_init(ompt_fns_t *fns) {
  // no initialization of OMPT for libomptarget unless 
  // libomp implements this function
  DP("in dummy libomp_libomptarget_ompt_init\n");
}


static void libomptarget_ompt_initialize(ompt_function_lookup_t lookup, ompt_fns_t *fns) {
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
}


static void libomptarget_ompt_finalize(ompt_fns_t *fns) {
  DP("enter libomptarget_ompt_finalize!\n");

  libomptarget_rtl_fns.finalize(); 

  ompt_enabled = false;

  DP("exit libomptarget_ompt_finalize!\n");
}


void ompt_init() {
  static ompt_fns_t libomptarget_ompt_fns;
  static bool initialized = false;

  if (initialized == false) {
    libomptarget_ompt_fns.initialize = libomptarget_ompt_initialize;
    libomptarget_ompt_fns.finalize   = libomptarget_ompt_finalize;
    
    DP("in ompt_init\n");
    libomp_libomptarget_ompt_init_t libomp_libomptarget_ompt_init_fn = 
      (libomp_libomptarget_ompt_init_t) (uint64_t) dlsym(NULL, "libomp_libomptarget_ompt_init");

    if (libomp_libomptarget_ompt_init_fn) {
      libomp_libomptarget_ompt_init_fn(&libomptarget_ompt_fns);
    }
    initialized = true;
  }
}

static void libomptarget_get_target_info(ompt_target_id_t *target_region_opid) {
  *target_region_opid = ompt_target_region_opid;
}

static ompt_interface_fn_t libomptarget_rtl_fn_lookup(const char *fname) {
  if (strcmp(fname, "libomptarget_get_target_info") == 0) 
    return (ompt_interface_fn_t) libomptarget_get_target_info;

#define lookup_libomp_fn(fn) \
  if (strcmp(fname, #fn) == 0) return (ompt_interface_fn_t) fn ## _fn;

  FOREACH_OMPT_TARGET_CALLBACK(lookup_libomp_fn)

#undef lookup_libomp_fn

  return 0;
}

extern "C" {

void
libomptarget_rtl_ompt_init
(
 ompt_fns_t *fns
)
{
  DP("enter libomptarget_rtl_ompt_init\n");
  if (ompt_enabled && fns) {
    libomptarget_rtl_fns.register_rtl(fns); 
    
    fns->initialize(libomptarget_rtl_fn_lookup, fns);
  }
  DP("leave libomptarget_rtl_ompt_init\n");
}

void
libomptarget_rtl_start_tool
(
 ompt_initialize_t target_rtl_init
)
{
  DP("in libomptarget_start_tool\n");
  if (ompt_enabled) {
    DP("calling target_rtl_init \n");
    target_rtl_init(libomptarget_rtl_fn_lookup, libomp_version_string, libomp_version_number);
  }
}

}
