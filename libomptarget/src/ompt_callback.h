//===----------- device.h - Target independent OpenMP target RTL ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declarations for OpenMP Tool callback dispatchers
//
//===----------------------------------------------------------------------===//

#ifndef _OMPTARGET_CALLBACK_H
#define _OMPTARGET_CALLBACK_H

struct OmptCallback {
  void *_codeptr;

  explicit OmptCallback(void *codeptr_ra) : _codeptr(codeptr_ra) {}

  // target op callbacks
  void target_data_alloc(void *TgtPtrBegin, size_t size);

  void target_data_submit(void *HstPtrBegin, void *TgtPtrBegin, size_t size);

  void target_data_delete(void *TgtPtrBegin); 

  void target_data_retrieve(void *HstPtrBegin, void *TgtPtrBegin, size_t size); 

  void target_submit();

  // target region callbacks
  void target_enter_data();

  void target_exit_data();

  void target_update();

  void target();

  // begin/end target region marks
  void target_region_begin();
  
  void target_region_end();

  // begin/end target op marks
  void OmptCallback::target_operation_begin();

  void OmptCallback::target_operation_end();
}; 

extern void ompt_init();

extern void libomptarget_ompt_initialize(ompt_function_lookup_t lookup, ompt_fns_t *fns);

extern void libomptarget_get_target_info(ompt_target_id_t *target_region_opid);

#endif
