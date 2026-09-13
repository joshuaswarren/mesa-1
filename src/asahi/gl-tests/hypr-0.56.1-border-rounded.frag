#version 300 es

#extension GL_ARB_shading_language_include : enable

precision highp float;
in vec2 v_texcoord;

uniform int sourceTF;
uniform int targetTF;
uniform mat3 targetPrimariesXYZ;

uniform vec2 fullSizeUntransformed;
uniform float radiusOuter;
uniform float thick;


uniform vec4 gradient[10];
uniform vec4 gradient2[10];
uniform int gradientLength;
uniform int gradient2Length;
uniform float angle;
uniform float angle2;
uniform float gradientLerp;
uniform float alpha;

uniform float radius;
uniform float roundingPower;
uniform vec2 topLeft;
uniform vec2 fullSize;




















float distanceWithRounding(vec2 coords, float roundingPower) {
    return pow(pow(coords.x, roundingPower) + pow(coords.y, roundingPower), 1.0 / roundingPower);
}

vec4 rounding(vec4 color, float radius, float roundingPower, vec2 topLeft, vec2 fullSize) {
    vec2 pixCoord = vec2(gl_FragCoord);
    pixCoord -= topLeft + fullSize * 0.5;
    pixCoord *= vec2(lessThan(pixCoord, vec2(0.0))) * - 2.0 + 1.0;
    pixCoord -= fullSize * 0.5 - radius;
    pixCoord += vec2(1.0, 1.0) / fullSize;

    if (pixCoord.x + pixCoord.y > radius) {
        float dist = distanceWithRounding(pixCoord, roundingPower);

        if (dist > radius + (3.1415926535897932384626433832795 / 5.34665792551))
            discard;

        float normalized = 1.0 - smoothstep(0.0, 1.0, (dist - radius + (3.1415926535897932384626433832795 / 5.34665792551)) / ( (3.1415926535897932384626433832795 / 5.34665792551) * 2.0));

        color *= normalized;
    }

    return color;
}


































































































vec3 applyIcc3DLut(vec3 linearRgb01, highp sampler3D iccLut3D, float iccLutSize) {
    vec3 x = clamp(linearRgb01, 0.0, 1.0);


    float N = iccLutSize;
    vec3 coord = (x * (N - 1.0) + 0.5) / N;

    return texture(iccLut3D, coord).rgb;
}

vec3 xy2xyz(vec2 xy) {
    if (xy.y == 0.0)
        return vec3(0.0, 0.0, 0.0);

    return vec3(xy.x / xy.y, 1.0, (1.0 - xy.x - xy.y) / xy.y);
}


vec3 tfInvPQ(vec3 color) {
    vec3 E = pow(clamp(color.rgb, vec3(0.0), vec3(1.0)), vec3( (1.0 / 78.84375)));
    return pow( (max(E - 0.8359375, vec3(0.0))) / (18.8515625 - 18.6875 * E), vec3( (1.0 / 0.1593017578125)));
}

vec3 tfInvHLG(vec3 color) {
    bvec3 isLow = lessThanEqual(color.rgb, vec3(0.5));
    vec3 lo = color.rgb * color.rgb / 3.0;
    vec3 hi = (exp( (color.rgb - 0.55991073) / 0.17883277) + 0.28466892) / 12.0;
    return mix(hi, lo, isLow);
}





vec3 tfInvLinPow(vec3 color, float gamma, float thres, float scale, float alpha) {
    bvec3 isLow = lessThanEqual(color.rgb, vec3(thres * scale));
    vec3 lo = color.rgb / scale;
    vec3 hi = pow( (color.rgb + alpha - 1.0) / alpha, vec3(gamma));
    return mix(hi, lo, isLow);
}

vec3 tfInvSRGB(vec3 color) {
    return tfInvLinPow(color, 2.4, 0.0031308, 12.92, 1.055);
}

vec3 tfInvExtSRGB(vec3 color) {

    return sign(color) * tfInvSRGB(abs(color));
}

vec3 tfInvBT1886(vec3 color) {
    return tfInvLinPow(color, (1.0 / 0.45), 0.018053968510807, 4.5, (1.0 + 5.5 * 0.018053968510807));
}

vec3 tfInvXVYCC(vec3 color) {


    return sign(color) * tfInvBT1886(abs(color));
}

vec3 tfInvST240(vec3 color) {
    return tfInvLinPow(color, (1.0 / 0.45), 0.0228, 4.0, 1.1115);
}


vec3 tfPQ(vec3 color) {
    vec3 E = pow(clamp(color.rgb, vec3(0.0), vec3(1.0)), vec3(0.1593017578125));
    return pow( (vec3(0.8359375) + 18.8515625 * E) / (vec3(1.0) + 18.6875 * E), vec3(78.84375));
}

vec3 tfHLG(vec3 color) {
    bvec3 isLow = lessThanEqual(color.rgb, vec3( (1.0 / 12.0)));
    vec3 lo = sqrt(max(color.rgb, vec3(0.0)) * 3.0);
    vec3 hi = 0.17883277 * log(max(12.0 * color.rgb - 0.28466892, vec3(0.0001))) + 0.55991073;
    return mix(hi, lo, isLow);
}

vec3 tfLinPow(vec3 color, float gamma, float thres, float scale, float alpha) {
    bvec3 isLow = lessThanEqual(color.rgb, vec3(thres));
    vec3 lo = color.rgb * scale;
    vec3 hi = pow(color.rgb, vec3(1.0 / gamma)) * alpha - (alpha - 1.0);
    return mix(hi, lo, isLow);
}

vec3 tfSRGB(vec3 color) {
    return tfLinPow(color, 2.4, 0.0031308, 12.92, 1.055);
}

vec3 tfExtSRGB(vec3 color) {

    return sign(color) * tfSRGB(abs(color));
}

vec3 tfBT1886(vec3 color) {
    return tfLinPow(color, (1.0 / 0.45), 0.018053968510807, 4.5, (1.0 + 5.5 * 0.018053968510807));
}

vec3 tfXVYCC(vec3 color) {


    return sign(color) * tfBT1886(abs(color));
}

vec3 tfST240(vec3 color) {
    return tfLinPow(color, (1.0 / 0.45), 0.0228, 4.0, 1.1115);
}

vec3 toLinearRGB(vec3 color, int tf) {
    switch (tf) {
        case 0 : return color;
        case 5 : return color;
        case 11 : return tfInvPQ(color);
        case 2 : return pow(max(color, vec3(0.0)), vec3(2.2));
        case 3 : return pow(max(color, vec3(0.0)), vec3(2.8));
        case 13 : return tfInvHLG(color);
        case 10 : return tfInvExtSRGB(color);
        case 1 : return tfInvBT1886(color);
        case 4 : return tfInvST240(color);
        case 6 : return mix(exp( (color - 1.0) * 2.0 * log(10.0)), vec3(0.0), lessThanEqual(color, vec3(0.0)));
        case 7 : return mix(exp( (color - 1.0) * 2.5 * log(10.0)), vec3(0.0), lessThanEqual(color, vec3(0.0)));
        case 8 : return tfInvXVYCC(color);
        case 12 : return pow(max(color, vec3(0.0)), vec3(2.6)) * (52.37 / 48.0);
        case 9 :
        default : return tfInvSRGB(color);
    }
}

vec4 toLinear(vec4 color, int tf) {
    if (tf == 5)
        return color;

    color.rgb /= max(color.a, 0.001);
    color.rgb = toLinearRGB(color.rgb, tf);
    color.rgb *= color.a;
    return color;
}

vec4 toNit(vec4 color, vec2 range) {
    color.rgb = color.rgb * (range[1] - range[0]) + range[0];
    return color;
}

vec3 fromLinearRGB(vec3 color, int tf) {
    switch (tf) {
        case 5 : return color;
        case 11 : return tfPQ(color);
        case 2 : return pow(max(color, vec3(0.0)), vec3(1.0 / 2.2));
        case 3 : return pow(max(color, vec3(0.0)), vec3(1.0 / 2.8));
        case 13 : return tfHLG(color);
        case 10 : return tfExtSRGB(color);
        case 1 : return tfBT1886(color);
        case 4 : return tfST240(color);
        case 6 : return mix(1.0 + log(color) / log(10.0) / 2.0, vec3(0.0), lessThanEqual(color, vec3(0.01)));
        case 7 : return mix(1.0 + log(color) / log(10.0) / 2.5, vec3(0.0), lessThanEqual(color, vec3(sqrt(10.0) / 1000.0)));
        case 8 : return tfXVYCC(color);
        case 12 : return pow(max(color, vec3(0.0)) / (52.37 / 48.0), vec3(1.0 / 2.6));
        case 9 :
        default : return tfSRGB(color);
    }
}

vec4 fromLinear(vec4 color, int tf) {
    if (tf == 5 || tf == 0)
        return color;

    color.rgb /= max(color.a, 0.001);
    color.rgb = fromLinearRGB(color.rgb, tf);
    color.rgb *= color.a;
    return color;
}

vec4 fromLinearNit(vec4 color, int tf, vec2 range) {
    if (tf == 0)
        return color;

    color.rgb = (color.rgb - range[0] * color.a) / (range[1] - range[0]);
    color.rgb /= max(color.a, 0.001);
    color.rgb = fromLinearRGB(color.rgb, tf);
    color.rgb *= color.a;
    return color;
}








vec4

    doColorManagement(vec4 pixColor, float additionalAlpha, int srcTF, int dstTF, mat3 convertMatrix, vec2 srcTFRange, vec2 dstTFRange

















    ) {
    float sourceAlpha = pixColor.a;
    float finalAlpha = sourceAlpha * additionalAlpha;

    pixColor.rgb /= max(sourceAlpha, 0.001);
    pixColor.rgb = toLinearRGB(pixColor.rgb, srcTF);






    pixColor.rgb = convertMatrix * pixColor.rgb;
    if (srcTF != 0)
        pixColor = toNit(pixColor, srcTFRange);
    pixColor.a = finalAlpha;
    pixColor.rgb *= pixColor.a;









    pixColor = fromLinearNit(pixColor, dstTF, dstTFRange);












    return pixColor;

}



uniform vec2 srcTFRange;
uniform vec2 dstTFRange;

uniform float srcRefLuminance;
uniform mat3 convertMatrix;













































































































































































































































































































vec4 okLabAToSrgb(vec4 lab) {
    float l = pow(lab[0] + lab[1] * 0.3963377774 + lab[2] * 0.2158037573, 3.0);
    float m = pow(lab[0] + lab[1] * (- 0.1055613458) + lab[2] * (- 0.0638541728), 3.0);
    float s = pow(lab[0] + lab[1] * (- 0.0894841775) + lab[2] * (- 1.2914855480), 3.0);

    return vec4(fromLinearRGB(vec3(l * 4.0767416621 + m * - 3.3077115913 + s * 0.2309699292, l * (- 1.2684380046) + m * 2.6097574011 + s * (- 0.3413193965),
                                   l * (- 0.0041960863) + m * (- 0.7034186147) + s * 1.7076147010),
                                                         2),
                lab[3]);
}

vec4 getOkColorForCoordArray1(vec2 normalizedCoord, int gradientLength, vec4 gradient[10], float angle) {
    if (gradientLength <= 0)
        return vec4(0.0);
    if (gradientLength < 2)
        return gradient[0];

    float finalAng = 0.0;

    if (angle > 4.71) {
        normalizedCoord[1] = 1.0 - normalizedCoord[1];
        finalAng = 6.28 - angle;
    } else if (angle > 3.14) {
        normalizedCoord[0] = 1.0 - normalizedCoord[0];
        normalizedCoord[1] = 1.0 - normalizedCoord[1];
        finalAng = angle - 3.14;
    } else if (angle > 1.57) {
        normalizedCoord[0] = 1.0 - normalizedCoord[0];
        finalAng = 3.14 - angle;
    } else {
        finalAng = angle;
    }

    float sine = sin(finalAng);
    float progress = (normalizedCoord[1] * sine + normalizedCoord[0] * (1.0 - sine)) * float(gradientLength - 1);

    if (progress >= float(gradientLength - 1))
        return gradient[gradientLength - 1];

    int bottom = int(floor(progress));
    int top = bottom + 1;

    return gradient[top] * (progress - float(bottom)) + gradient[bottom] * (float(top) - progress);
}

vec4 getOkColorForCoordArray2(vec2 normalizedCoord, int gradient2Length, vec4 gradient2[10], float angle2) {
    if (gradient2Length < 2)
        return gradient2[0];

    float finalAng = 0.0;

    if (angle2 > 4.71) {
        normalizedCoord[1] = 1.0 - normalizedCoord[1];
        finalAng = 6.28 - angle2;
    } else if (angle2 > 3.14) {
        normalizedCoord[0] = 1.0 - normalizedCoord[0];
        normalizedCoord[1] = 1.0 - normalizedCoord[1];
        finalAng = angle2 - 3.14;
    } else if (angle2 > 1.57) {
        normalizedCoord[0] = 1.0 - normalizedCoord[0];
        finalAng = 3.14 - angle2;
    } else {
        finalAng = angle2;
    }

    float sine = sin(finalAng);
    float progress = (normalizedCoord[1] * sine + normalizedCoord[0] * (1.0 - sine)) * float(gradient2Length - 1);

    if (progress >= float(gradient2Length - 1))
        return gradient2[gradient2Length - 1];

    int bottom = int(floor(progress));
    int top = bottom + 1;

    return gradient2[top] * (progress - float(bottom)) + gradient2[bottom] * (float(top) - progress);
}

vec4 getColorForCoord(vec2 normalizedCoord, int gradientLength, vec4 gradient[10], float angle, int gradient2Length, vec4 gradient2[10], float angle2, float gradientLerp) {
    vec4 result1 = getOkColorForCoordArray1(normalizedCoord, gradientLength, gradient, angle);

    if (gradient2Length <= 0)
        return okLabAToSrgb(result1);

    vec4 result2 = getOkColorForCoordArray2(normalizedCoord, gradient2Length, gradient2, angle2);

    return okLabAToSrgb(mix(result1, result2, gradientLerp));
}






































vec4

    getBorder(vec2 v_texcoord, float alpha, vec2 fullSizeUntransformed, float radiusOuter, float thick, float radius, float roundingPower, vec2 topLeft, vec2 fullSize,
              int gradientLength, vec4 gradient[10], float angle, int gradient2Length, vec4 gradient2[10], float angle2, float gradientLerp





















    ) {
    vec2 pixCoord = vec2(gl_FragCoord);
    vec2 pixCoordOuter = pixCoord;
    vec2 originalPixCoord = v_texcoord;
    originalPixCoord *= fullSizeUntransformed;
    float additionalAlpha = 1.0;

    vec4 pixColor = vec4(1.0, 1.0, 1.0, 1.0);

    bool done = false;

    pixCoord -= topLeft + fullSize * 0.5;
    pixCoord *= vec2(lessThan(pixCoord, vec2(0.0))) * - 2.0 + 1.0;
    pixCoordOuter = pixCoord;
    pixCoord -= fullSize * 0.5 - radius;
    pixCoordOuter -= fullSize * 0.5 - radiusOuter;


    pixCoord += vec2(1.0, 1.0) / fullSize;
    pixCoordOuter += vec2(1.0, 1.0) / fullSize;


    if (min(pixCoord.x, pixCoord.y) > 0.0 && radius > 0.0) {
        float dist = pow(pow(pixCoord.x, roundingPower) + pow(pixCoord.y, roundingPower), 1.0 / roundingPower);
        float distOuter = pow(pow(pixCoordOuter.x, roundingPower) + pow(pixCoordOuter.y, roundingPower), 1.0 / roundingPower);
        float h = (thick / 2.0);

        if (dist < radius - h) {

            float normalized = smoothstep(0.0, 1.0, (dist - radius + thick + (3.1415926535897932384626433832795 / 5.34665792551)) / ( (3.1415926535897932384626433832795 / 5.34665792551) * 2.0));
            additionalAlpha *= normalized;
            done = true;
        } else if (min(pixCoordOuter.x, pixCoordOuter.y) > 0.0) {

            float normalized = 1.0 - smoothstep(0.0, 1.0, (distOuter - radiusOuter + (3.1415926535897932384626433832795 / 5.34665792551)) / ( (3.1415926535897932384626433832795 / 5.34665792551) * 2.0));
            additionalAlpha *= normalized;
            done = true;
        } else if (distOuter < radiusOuter - h) {
            additionalAlpha = 1.0;
            done = true;
        }
    }



    if (! done) {

        float distanceT = originalPixCoord[1];
        float distanceB = fullSizeUntransformed[1] - originalPixCoord[1];
        float distanceL = originalPixCoord[0];
        float distanceR = fullSizeUntransformed[0] - originalPixCoord[0];


        float smallest = min(min(distanceT, distanceB), min(distanceL, distanceR));

        if (smallest > thick)
            discard;
    }

    if (additionalAlpha == 0.0)
        discard;

    pixColor = getColorForCoord(v_texcoord, gradientLength, gradient, angle, gradient2Length, gradient2, angle2, gradientLerp);
    pixColor.rgb *= pixColor[3];





























    pixColor *= alpha * additionalAlpha;
    return pixColor;

}

layout(location = 0) out vec4 fragColor;



void main() {



    fragColor =

        getBorder(v_texcoord, alpha, fullSizeUntransformed, radiusOuter, thick, radius, roundingPower, topLeft, fullSize, gradientLength, gradient, angle, gradient2Length,
                  gradient2, angle2, gradientLerp





















        );




}
