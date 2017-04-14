#ifndef OMPTARGET_OMPT_H
#define OMPTARGET_OMPT_H

#include <ompt.h>

#define FOREACH_OMPT_TARGET_FN(macro)		 \
  macro (libomp_callback_device_initialize)	 \
  macro (libomp_callback_device_finalize)	 \
  macro (libomp_callback_device_load)		 \
  macro (libomp_callback_device_unload)		 \
  macro (libomp_callback_target)		 \
  macro (libomp_set_frame_reenter)      


typedef ompt_callback_device_initialize_t libomp_callback_device_initialize_t;
typedef ompt_callback_device_finalize_t   libomp_callback_device_finalize_t;

typedef ompt_callback_device_load_t       libomp_callback_device_load_t;
typedef ompt_callback_device_unload_t     libomp_callback_device_unload_t;
typedef ompt_callback_device_finalize_t   libomp_callback_device_finalize_t;

typedef ompt_callback_target_t libomp_callback_target_t;

typedef void (*libomp_set_frame_reenter_t)
(
 void *addr
);



#endif /* OMPTARGET_OMPT_H */
