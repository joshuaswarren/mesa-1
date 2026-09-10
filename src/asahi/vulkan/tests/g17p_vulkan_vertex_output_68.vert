#version 450

layout(location = 0) in vec2 position;
layout(location = 0) out vec4 user_output[17];

void
main()
{
   gl_Position = vec4(position, 0.0, 1.0);

   /* Keep every component live across the stage interface. All three
    * full-screen-triangle vertices have x+y equal to -2 or +2, so
    * this is exactly zero at every vertex and remains zero after interpolation.
    * It still depends on runtime vertex-buffer data, preventing the linker from
    * replacing the 68-component interface with compile-time constants. */
   float vertex = (position.x + position.y) * (position.x + position.y) - 4.0;
   vec4 value0 = vec4(0.0, 1.0, 2.0, 3.0) +
                    vertex * vec4(1.0, 2.0, 3.0, 4.0) / 1024.0;
   vec4 value1 = vec4(4.0, 5.0, 6.0, 7.0) +
                    vertex * vec4(5.0, 6.0, 7.0, 8.0) / 1024.0;
   vec4 value2 = vec4(8.0, 9.0, 10.0, 11.0) +
                    vertex * vec4(9.0, 10.0, 11.0, 12.0) / 1024.0;
   vec4 value3 = vec4(12.0, 13.0, 14.0, 15.0) +
                    vertex * vec4(13.0, 14.0, 15.0, 16.0) / 1024.0;
   vec4 value4 = vec4(16.0, 17.0, 18.0, 19.0) +
                    vertex * vec4(17.0, 18.0, 19.0, 20.0) / 1024.0;
   vec4 value5 = vec4(20.0, 21.0, 22.0, 23.0) +
                    vertex * vec4(21.0, 22.0, 23.0, 24.0) / 1024.0;
   vec4 value6 = vec4(24.0, 25.0, 26.0, 27.0) +
                    vertex * vec4(25.0, 26.0, 27.0, 28.0) / 1024.0;
   vec4 value7 = vec4(28.0, 29.0, 30.0, 31.0) +
                    vertex * vec4(29.0, 30.0, 31.0, 32.0) / 1024.0;
   vec4 value8 = vec4(32.0, 33.0, 34.0, 35.0) +
                    vertex * vec4(33.0, 34.0, 35.0, 36.0) / 1024.0;
   vec4 value9 = vec4(36.0, 37.0, 38.0, 39.0) +
                    vertex * vec4(37.0, 38.0, 39.0, 40.0) / 1024.0;
   vec4 value10 = vec4(40.0, 41.0, 42.0, 43.0) +
                     vertex * vec4(41.0, 42.0, 43.0, 44.0) / 1024.0;
   vec4 value11 = vec4(44.0, 45.0, 46.0, 47.0) +
                     vertex * vec4(45.0, 46.0, 47.0, 48.0) / 1024.0;
   vec4 value12 = vec4(48.0, 49.0, 50.0, 51.0) +
                     vertex * vec4(49.0, 50.0, 51.0, 52.0) / 1024.0;
   vec4 value13 = vec4(52.0, 53.0, 54.0, 55.0) +
                     vertex * vec4(53.0, 54.0, 55.0, 56.0) / 1024.0;
   vec4 value14 = vec4(56.0, 57.0, 58.0, 59.0) +
                     vertex * vec4(57.0, 58.0, 59.0, 60.0) / 1024.0;
   vec4 value15 = vec4(60.0, 61.0, 62.0, 63.0) +
                     vertex * vec4(61.0, 62.0, 63.0, 64.0) / 1024.0;
   vec4 value16 = vec4(64.0, 65.0, 66.0, 67.0) +
                     vertex * vec4(65.0, 66.0, 67.0, 68.0) / 1024.0;

   /* Terminal stores select the driver's qualified batched UVS protocol. */
   user_output[0] = value0;
   user_output[1] = value1;
   user_output[2] = value2;
   user_output[3] = value3;
   user_output[4] = value4;
   user_output[5] = value5;
   user_output[6] = value6;
   user_output[7] = value7;
   user_output[8] = value8;
   user_output[9] = value9;
   user_output[10] = value10;
   user_output[11] = value11;
   user_output[12] = value12;
   user_output[13] = value13;
   user_output[14] = value14;
   user_output[15] = value15;
   user_output[16] = value16;
}
