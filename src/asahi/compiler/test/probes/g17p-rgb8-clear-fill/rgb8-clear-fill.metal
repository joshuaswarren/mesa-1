#include <metal_stdlib>
using namespace metal;

kernel void
fill_rgb8_padded(device uchar4 *destination [[buffer(0)]],
                 constant uchar3 &value [[buffer(1)]],
                 uint index [[thread_position_in_grid]])
{
   destination[index] = uchar4(value.x, value.y, value.z, 255);
}
