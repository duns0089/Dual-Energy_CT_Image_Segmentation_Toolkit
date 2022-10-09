#pragma once
#ifndef HELPERS_H
#define HELPERS_H

#pragma warning(push, 0) // suppress 3rd party header warnings
#include <cuda_runtime.h>
#include <helper_math.h>
#pragma warning(pop) // stop suppression of warnings


// Inline functions
inline __host__ __device__ int size(int3 v) {
    return (int)(v.x * v.y * v.z);
}
inline __host__ __device__ int size(int2 v) {
    return (int)(v.x * v.y);
}
inline __host__ __device__ void operator/=(int3& a, int b) {
    a.x /= b;
    a.y /= b;
    a.z /= b;
}
inline __host__ __device__ void operator/=(int2& a, int b) {
    a.x /= b;
    a.y /= b;
}
inline __host__ __device__ void operator/=(float2& a, int b) {
    a.x /= b;
    a.y /= b;
}
inline __host__ __device__ float3 operator*(int3 a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ float3 operator*(float b, int3 a) {
    return make_float3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ float3 operator*(float3 a, int3 b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ float3 operator*(int3 a, float3 b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(float2& a, int b) {
    a.x *= b;
    a.y *= b;
}
// Rotation in direction about an axis
inline __host__ __device__ float3 rotateCCW_X(float3 a, float radians) {
    return make_float3(a.x, a.y * cosf(radians) - a.z * sinf(radians), a.y * sinf(radians) + a.z * cosf(radians));
}
inline __host__ __device__ float3 rotateCW_X(float3 a, float radians) {
    return make_float3(a.x, a.y * cosf(radians) + a.z * sinf(radians), -a.y * sinf(radians) + a.z * cosf(radians));
}
inline __host__ __device__ float3 rotateCCW_Y(float3 a, float radians) {
    return make_float3(a.x * cosf(radians) - a.z * sinf(radians), a.y, a.x * sinf(radians) + a.z * cosf(radians));
}
inline __host__ __device__ float3 rotateCW_Y(float3 a, float radians) {
    return make_float3(a.x * cosf(radians) + a.z * sinf(radians), a.y, -a.x * sinf(radians) + a.z * cosf(radians));
}
inline __host__ __device__ float3 rotateCW_Z(float3 a, float radians) {
    return make_float3(a.x * cosf(radians) + a.y * sinf(radians), - a.x * sinf(radians) + a.y * cosf(radians), a.z);
}
inline __host__ __device__ float3 rotateCCW_Z(float3 a, float radians) {
    return make_float3(a.x * cosf(radians) - a.y * sinf(radians), a.x * sinf(radians) + a.y * cosf(radians), a.z);
    //return make_float3(a.x * cosf(radians) + a.y * sinf(radians), -a.x * sinf(radians) + a.y * cosf(radians), a.z);
}
inline __host__ __device__ float2 rotateCW_Z(float2 a, float radians) {
    return make_float2(a.x * cosf(radians) + a.y * sinf(radians), a.y * cosf(radians) - a.x * sinf(radians));
}
inline __host__ __device__ float2 rotateCCW_Z(float2 a, float radians) {
    return make_float2(a.x * cosf(radians) - a.y * sinf(radians), a.x * sinf(radians) + a.y * cosf(radians));
}

// Array indexing macros
#define INDEX2D(n, m, n_length) ((n) + (m) * (n_length))												//!< 2D to 1D array index arithmetic
#define INDEX3D(n, m, k, n_length, m_length) ((n) + (m) * (n_length) + (k) * (n_length) * (m_length))	//!< 3D to 1D array index arithmetic

#define POW2(x) (x)*(x) //!< Square a number

const float PI_FLOAT = 3.14159265359f;
const float RADIANS = PI_FLOAT / 180.0f;
const float DEGREES = 180.0f / PI_FLOAT;

#ifndef M_PI
/// Portable definition of M_PI
#define M_PI 3.14159265358979323846
#endif

// Math Conversions
#ifndef DEG2RAD
#define DEG2RAD (float)M_PI/180.0f //!< Convert Degrees to Radians in float
#endif
#ifndef RAD2DEG 
#define RAD2DEG 180.0f/(float)M_PI //!< Convert Radians to Degrees in float  
#endif

#ifndef INVCM2_TO_INVMM2
#define INVCM2_TO_INVMM2 1.0f/100.0f //!< Convert cm^-2 to mm^-2
#endif // !INVCM2_TO_INVMM2

#ifndef MEV_TO_KEV
#define MEV_TO_KEV 1000.0f //!< Convert MeV to keV
#endif //!MEV_TO_KEV

#ifndef KEV_TO_MEV
#define KEV_TO_MEV 1.0f/1000.0f //!< Convert keV to MeV
#endif // !KEV_TO_MEV

#ifndef MM3_TO_CM3
#define MM3_TO_CM3 1.0f/1000.0f //!< Convert millimeters cubed to centimeters cubed
#endif // !MM3_TO_CM3

#ifndef MM_TO_CM
#define MM_TO_CM 1.0f/10.0f //!< Convert millimeters to centimeters
#endif // !MM_TO_CM

#ifndef S_TO_MS
#define S_TO_MS 1000.0f //!< Convert seconds to milliseconds
#endif // !S_TO_MS

#ifndef MS_TO_S
#define MS_TO_S 1.0f/1000.0f //!< Convert milliseconds to seconds
#endif // !MS_TO_S

#ifndef MIN_TO_S
#define MIN_TO_S 60.0f //!< Convert minutes to seconds
#endif // !MIN_TO_S

#ifndef S_TO_MIN
#define S_TO_MIN 1.0f/60.0f //!< Convert seconds to minutes
#endif // !S_TO_MIN

#ifndef HR_TO_MIN
#define HR_TO_MIN 60.0f //!< Convert hours to minutes
#endif // !HR_TO_MIN

#ifndef MIN_TO_HR
#define MIN_TO_HR 1.0f/60.0f //!< Convert minutes to hours
#endif // !MIN_TO_HR

#ifndef HR_TO_S
#define HR_TO_S (HR_TO_MIN * MIN_TO_S) //!< Convert hours to seconds
#endif // !HR_TO_S

#ifndef S_TO_HR
#define S_TO_HR (S_TO_MIN * MIN_TO_HR) //!< Convert seconds to hours
#endif // !S_TO_HR

#ifndef HR_TO_MS
#define HR_TO_MS (HR_TO_S * S_TO_MS) //!< Convert hours to milliseconds
#endif // !HR_TO_MS

#ifndef MS_TO_HR
#define MS_TO_HR (MS_TO_S * S_TO_HR) //!< Convert milliseconds to hours
#endif // !MS_TO_HR

#ifndef MIN_TO_MS
#define MIN_TO_MS (MIN_TO_S * S_TO_MS) //!< Convert minutes to milliseconds
#endif // !MIN_TO_MS

#ifndef MS_TO_MIN
#define MS_TO_MIN (MS_TO_S * S_TO_MIN) //!< Convert milliseconds to minutes
#endif // !MS_TO_MIN

#ifndef FLT_EPSILON
#define FLT_EPSILON      1.192092896e-07F        // smallest such that 1.0+FLT_EPSILON != 1.0
#endif // !FLT_EPSILON



inline __host__ __device__ float fx(float A, float B, float C, float x) {
    return -(A * x + C) / B;
};
inline __host__ __device__ float fy(float A, float B, float C, float y) {
    return -(B * y + C) / A;
};


#endif // !HELPERS_H
