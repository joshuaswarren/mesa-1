#version 450

layout(location = 0) in vec4 user_input[17];
layout(location = 0) out vec4 out_color;

void
main()
{
   /* Squaring after interpolation prevents a legal inter-stage optimization
    * from replacing these 68 scalar inputs with one pre-summed varying. */
   float checksum = dot(user_input[0], user_input[0]);
   checksum += dot(user_input[1], user_input[1]);
   checksum += dot(user_input[2], user_input[2]);
   checksum += dot(user_input[3], user_input[3]);
   checksum += dot(user_input[4], user_input[4]);
   checksum += dot(user_input[5], user_input[5]);
   checksum += dot(user_input[6], user_input[6]);
   checksum += dot(user_input[7], user_input[7]);
   checksum += dot(user_input[8], user_input[8]);
   checksum += dot(user_input[9], user_input[9]);
   checksum += dot(user_input[10], user_input[10]);
   checksum += dot(user_input[11], user_input[11]);
   checksum += dot(user_input[12], user_input[12]);
   checksum += dot(user_input[13], user_input[13]);
   checksum += dot(user_input[14], user_input[14]);
   checksum += dot(user_input[15], user_input[15]);
   checksum += dot(user_input[16], user_input[16]);

   /* 0^2 + ... + 67^2 is the exactly representable integer 102510. */
   float mismatch = checksum - 102510.0;
   out_color = vec4(1.0, 0.28, 0.05, 1.0) + mismatch * (1.0 / 255.0);
}
