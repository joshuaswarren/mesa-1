/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT */
#include "d12_native_render_resources_core.h"
void
d12_native_render_resource_specs_pack(
   const struct d12_native_render_resource_specs *s,
   struct drm_asahi_cmd_render *c)
{
   c->bg.rsrc_spec = (uint32_t)s->bg;
   c->eot.rsrc_spec = (uint32_t)s->eot;
   c->partial_bg.rsrc_spec = (uint32_t)s->partial_bg;
   c->partial_eot.rsrc_spec = (uint32_t)s->partial_eot;
   c->bg_rsrc_spec_hi = (uint32_t)(s->bg >> 32);
   c->bg_eot_rsrc_spec_hi = (uint32_t)(s->eot >> 32);
   c->bg_partial_rsrc_spec_hi = (uint32_t)(s->partial_bg >> 32);
   c->bg_eot_partial_rsrc_spec_hi = (uint32_t)(s->partial_eot >> 32);
   if (c->bg_rsrc_spec_hi | c->bg_eot_rsrc_spec_hi |
       c->bg_partial_rsrc_spec_hi | c->bg_eot_partial_rsrc_spec_hi)
      c->flags |= DRM_ASAHI_RENDER_RSRC_SPEC_HI;
   else
      c->flags &= ~DRM_ASAHI_RENDER_RSRC_SPEC_HI;
}
