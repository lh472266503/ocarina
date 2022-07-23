#pragma once

using oc_int = int;
using oc_uint = unsigned int;
using oc_float = float;
using oc_bool = bool;

struct alignas(8) oc_int2{
	oc_int x;
	oc_int y;

	__device__ oc_int2() noexcept 
		:x{},y{} {}
	__device__ oc_int2(oc_int s) noexcept 
		:x(s),y(s) {}
};

struct alignas(8) oc_uint2{
	oc_uint x;
	oc_uint y;

	__device__ oc_uint2() noexcept 
		:x{},y{} {}
	__device__ oc_uint2(oc_uint s) noexcept 
		:x(s),y(s) {}
};

struct alignas(8) oc_float2{
	oc_float x;
	oc_float y;

	__device__ oc_float2() noexcept 
		:x{},y{} {}
	__device__ oc_float2(oc_float s) noexcept 
		:x(s),y(s) {}
};

struct alignas(8) oc_bool2{
	oc_bool x;
	oc_bool y;

	__device__ oc_bool2() noexcept 
		:x{},y{} {}
	__device__ oc_bool2(oc_bool s) noexcept 
		:x(s),y(s) {}
};

struct alignas(16) oc_int3{
	oc_int x;
	oc_int y;
	oc_int z;

	__device__ oc_int3() noexcept 
		:x{},y{},z{} {}
	__device__ oc_int3(oc_int s) noexcept 
		:x(s),y(s),z(s) {}
};

struct alignas(16) oc_uint3{
	oc_uint x;
	oc_uint y;
	oc_uint z;

	__device__ oc_uint3() noexcept 
		:x{},y{},z{} {}
	__device__ oc_uint3(oc_uint s) noexcept 
		:x(s),y(s),z(s) {}
};

struct alignas(16) oc_float3{
	oc_float x;
	oc_float y;
	oc_float z;

	__device__ oc_float3() noexcept 
		:x{},y{},z{} {}
	__device__ oc_float3(oc_float s) noexcept 
		:x(s),y(s),z(s) {}
};

struct alignas(16) oc_bool3{
	oc_bool x;
	oc_bool y;
	oc_bool z;

	__device__ oc_bool3() noexcept 
		:x{},y{},z{} {}
	__device__ oc_bool3(oc_bool s) noexcept 
		:x(s),y(s),z(s) {}
};

struct alignas(16) oc_int4{
	oc_int x;
	oc_int y;
	oc_int z;
	oc_int w;

	__device__ oc_int4() noexcept 
		:x{},y{},z{},w{} {}
	__device__ oc_int4(oc_int s) noexcept 
		:x(s),y(s),z(s),w(s) {}
};

struct alignas(16) oc_uint4{
	oc_uint x;
	oc_uint y;
	oc_uint z;
	oc_uint w;

	__device__ oc_uint4() noexcept 
		:x{},y{},z{},w{} {}
	__device__ oc_uint4(oc_uint s) noexcept 
		:x(s),y(s),z(s),w(s) {}
};

struct alignas(16) oc_float4{
	oc_float x;
	oc_float y;
	oc_float z;
	oc_float w;

	__device__ oc_float4() noexcept 
		:x{},y{},z{},w{} {}
	__device__ oc_float4(oc_float s) noexcept 
		:x(s),y(s),z(s),w(s) {}
};

struct alignas(16) oc_bool4{
	oc_bool x;
	oc_bool y;
	oc_bool z;
	oc_bool w;

	__device__ oc_bool4() noexcept 
		:x{},y{},z{},w{} {}
	__device__ oc_bool4(oc_bool s) noexcept 
		:x(s),y(s),z(s),w(s) {}
};

