`Mesa <https://mesa3d.org>`_ - The 3D Graphics Library
======================================================


The projects
------------

Part of the Omarchy on Apple Silicon ecosystem:

- `omarchy-mlx <https://github.com/joshuaswarren/omarchy-mlx>`_ — MLX on the Apple GPU under Linux, and Core ML models on the Neural Engine..
- `omarchy-ane <https://github.com/joshuaswarren/omarchy-ane>`_ — Linux driver for the Apple Neural Engine: DRM kernel modules and libane..
- `mil-hwx-compiler <https://github.com/joshuaswarren/mil-hwx-compiler>`_ — compiles Core ML MIL into ANE programs on Linux (M1/M2 targets)..
- `metal2vk <https://github.com/joshuaswarren/metal2vk>`_ — translates Metal compute kernels to Vulkan SPIR-V with open tools..
- `omarchy-cluster <https://github.com/joshuaswarren/omarchy-cluster>`_ — one local model split across Macs on macOS and Omarchy Linux..
- `omarchy-apple-dev <https://github.com/joshuaswarren/omarchy-apple-dev>`_ — build and ship iOS apps from Omarchy Linux, no Xcode or Mac..
- `omarchy-egpu <https://github.com/joshuaswarren/omarchy-egpu>`_ — eGPU setup, diagnostics and measured local-inference results on Omarchy Linux..
- `ane-linux-experiments <https://github.com/joshuaswarren/ane-linux-experiments>`_ — Neural Engine on Linux: verified primitives, receipt-backed macOS parity..
- `coreglass <https://github.com/joshuaswarren/coreglass>`_ — profiler that shows where local inference loses speed on Apple Silicon Linux..
- `omarchy-inference-fleet <https://github.com/joshuaswarren/omarchy-inference-fleet>`_ — reproducible, pinned single-node MLX inference baselines..
- `remnic <https://github.com/joshuaswarren/remnic>`_ — local-first memory and context for AI agents..
- Platform upstream: `omarchy-mac <https://github.com/omacom/omarchy-mac>`_ — Omarchy on Apple Silicon Macs.

Source
------

This repository lives at https://gitlab.freedesktop.org/mesa/mesa.
Other repositories are likely forks, and code found there is not supported.


Build & install
---------------

You can find more information in our documentation (`docs/install.rst
<https://docs.mesa3d.org/install.html>`_), but the recommended way is to use
Meson (`docs/meson.rst <https://docs.mesa3d.org/meson.html>`_):

.. code-block:: sh

  $ meson setup build
  $ ninja -C build/
  $ sudo ninja -C build/ install

Support
-------

Many Mesa devs hang on IRC; if you're not sure which channel is
appropriate, you should ask your question on `OFTC's #dri-devel
<irc://irc.oftc.net/dri-devel>`_, someone will redirect you if
necessary.
Remember that not everyone is in the same timezone as you, so it might
take a while before someone qualified sees your question.
To figure out who you're talking to, or which nick to ping for your
question, check out `Who's Who on IRC
<https://dri.freedesktop.org/wiki/WhosWho/>`_.

The next best option is to ask your question in an email to the
mailing lists: `mesa-dev\@lists.freedesktop.org
<https://lists.freedesktop.org/mailman/listinfo/mesa-dev>`_


Bug reports
-----------

If you think something isn't working properly, please file a bug report
(`docs/bugs.rst <https://docs.mesa3d.org/bugs.html>`_).


Contributing
------------

Contributions are welcome, and step-by-step instructions can be found in our
documentation (`docs/submittingpatches.rst
<https://docs.mesa3d.org/submittingpatches.html>`_).

Note that Mesa uses gitlab for patches submission, review and discussions.
