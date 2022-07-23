#pragma once

using oc_int = int;
using oc_uint = unsigned int;
using oc_float = float;
using oc_bool = bool;

struct alignas(8) oc_int2{
	oc_int x;
	oc_int y;
};

struct alignas(8) oc_uint2{
	oc_uint x;
	oc_uint y;
};

struct alignas(8) oc_float2{
	oc_float x;
	oc_float y;
};

struct alignas(8) oc_bool2{
	oc_bool x;
	oc_bool y;
};

struct alignas(16) oc_int3{
	oc_int x;
	oc_int y;
	oc_int z;
};

struct alignas(16) oc_uint3{
	oc_uint x;
	oc_uint y;
	oc_uint z;
};

struct alignas(16) oc_float3{
	oc_float x;
	oc_float y;
	oc_float z;
};

struct alignas(16) oc_bool3{
	oc_bool x;
	oc_bool y;
	oc_bool z;
};

struct alignas(16) oc_int4{
	oc_int x;
	oc_int y;
	oc_int z;
	oc_int w;
};

struct alignas(16) oc_uint4{
	oc_uint x;
	oc_uint y;
	oc_uint z;
	oc_uint w;
};

struct alignas(16) oc_float4{
	oc_float x;
	oc_float y;
	oc_float z;
	oc_float w;
};

struct alignas(16) oc_bool4{
	oc_bool x;
	oc_bool y;
	oc_bool z;
	oc_bool w;
};

