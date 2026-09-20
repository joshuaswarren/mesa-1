# jw16 (M1 Max T6001 G13C) sin-FTZ A/B window — Main-owned, /tmp/m1-gpu.lock inode-12

Prepared 2026-09-20 CPU-only. No GPU, vulkaninfo, service or system change
was run in preparation. llm-inference stop/start and lock handling: Main.

1. caps first (fork DFTZ32 declared on G13C?): 
   VK_DRIVER_FILES=/tmp/mesa-sin-ftz-ab/jw16-fckey-icd.json vulkaninfo > caps-fckey.txt
   grep -A14 VkPhysicalDeviceVulkan12Properties caps-fckey.txt
2. probe A/B (lock held outside; probe script takes no lock):
   MATHREPRO=/tmp/mesa-sin-ftz-ab/bin/mathrepro    VK_DRIVER_FILES=/tmp/mesa-sin-ftz-ab/jw16-fckey-icd.json    python3 probe/sin-denorm-probe.py jw16-fckey-icd.json jw16-fckey
   ... then same with jw16-e167-icd.json / tag jw16-e167
3. bounded CTS fp32 (needs deqp-vk — NOT installed on jw16, Main decides):
   deqp-vk --deqp-caselist-file=probe/fp32-caselist-1.4.6.txt
   (1995 lines = 1965 fp32 @CTS-main + 30 extensionless; 1.4.6 package variant = 1989)

Pins: PINS.sha256 (drivers differ from M2 shas — different build host, same
source commits 807d9d86828 / e1677564284; mathrepro sha identical to M2).
Native: agx_tests 61/61 on jw16. ldd: zero unresolved deps; GLIBC ceiling 2.38.
