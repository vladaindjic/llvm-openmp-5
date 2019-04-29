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
#define OMPT_FRAME_FLAGS (ompt_frame_runtime | ompt_frame_cfa)
#else
#define OMPT_GET_FRAME_ADDRESS(level) __builtin_frame_address(level)
#define OMPT_FRAME_FLAGS (ompt_frame_runtime | ompt_frame_framepointer)
#endif

#define OMPT_GET_RETURN_ADDRESS(level) __builtin_return_address(level)

#include <ompt.h>

struct OmptInterface {
  void *_enter_frame;
  void *_codeptr_ra;
  int _state;

  void ompt_state_set_helper(void *enter_frame, void *codeptr_ra, int flags, int state);

  void ompt_state_set(void *enter_frame, void *codeptr_ra);

  void ompt_state_clear();

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

extern thread_local OmptInterface ompt_interface; 


#endif
