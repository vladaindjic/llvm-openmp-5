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

#if (__PPC64__ | __arm__)
#define OMPT_GET_FRAME_ADDRESS(level) (*(void **)__builtin_frame_address(level))
#else
#define OMPT_GET_FRAME_ADDRESS(level) __builtin_frame_address(level)
#endif

#include <ompt.h>

struct OmptCallback {
  void *_codeptr;

  explicit OmptCallback(void *codeptr_ra) : _codeptr(codeptr_ra) {}

  // target op callbacks
  void target_data_alloc(int64_t device_id, void *TgtPtrBegin, size_t Size);

  void target_data_submit(int64_t device_id, void *HstPtrBegin, void *TgtPtrBegin, size_t Size);

  void target_data_delete(int64_t device_id, void *TgtPtrBegin); 

  void target_data_retrieve(int64_t device_id, void *HstPtrBegin, void *TgtPtrBegin, size_t Size); 

  void target_submit();

  // target region callbacks
  void target_enter_data(int64_t device_id);

  void target_exit_data(int64_t device_id);

  void target_update(int64_t device_id);

  void target(int64_t device_id);

  // begin/end target region marks
  uint64_t target_region_begin();
  
  uint64_t target_region_end();

  // begin/end target op marks
  void target_operation_begin();

  void target_operation_end();
}; 

extern void ompt_init();

#endif
