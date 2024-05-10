//
// Created by Zero on 2022/7/15.
//

#include "cuda_codegen.h"
#include "ast/expression.h"
#include "cuda_device.h"

namespace ocarina {

void CUDACodegen::visit(const CallExpr *expr) noexcept {
    auto emit_act_arguments = [this](const CallExpr *expr) {
        current_scratch() << "(";
        for (const auto &arg : expr->arguments()) {
            arg->accept(*this);
            current_scratch() << ",";
        }
        if (!expr->arguments().empty()) {
            current_scratch().pop_back();
        }
        current_scratch() << ")";
    };
    string_view func_name = expr->function_name();
    if (!func_name.empty()) {
        current_scratch() << func_name;
        emit_act_arguments(expr);
        return;
    }

#define OC_GEN_FUNC_NAME(func_name) current_scratch() << TYPE_PREFIX #func_name
    switch (expr->call_op()) {
        case CallOp::CUSTOM: CppCodegen::visit(expr); return;
        case CallOp::ALL: OC_GEN_FUNC_NAME(all); break;
        case CallOp::ANY: OC_GEN_FUNC_NAME(any); break;
        case CallOp::NONE: OC_GEN_FUNC_NAME(none); break;
        case CallOp::SELECT: OC_GEN_FUNC_NAME(select); break;
        case CallOp::CLAMP: OC_GEN_FUNC_NAME(clamp); break;
        case CallOp::LERP: OC_GEN_FUNC_NAME(lerp); break;
        case CallOp::ABS: OC_GEN_FUNC_NAME(abs); break;
        case CallOp::MIN: OC_GEN_FUNC_NAME(min); break;
        case CallOp::MAX: OC_GEN_FUNC_NAME(max); break;
        case CallOp::IS_INF: OC_GEN_FUNC_NAME(is_inf); break;
        case CallOp::IS_NAN: OC_GEN_FUNC_NAME(is_nan); break;
        case CallOp::ACOS: OC_GEN_FUNC_NAME(acos); break;
        case CallOp::ASIN: OC_GEN_FUNC_NAME(asin); break;
        case CallOp::ATAN: OC_GEN_FUNC_NAME(atan); break;
        case CallOp::ACOSH: OC_GEN_FUNC_NAME(acosh); break;
        case CallOp::ASINH: OC_GEN_FUNC_NAME(asinh); break;
        case CallOp::ATANH: OC_GEN_FUNC_NAME(atanh); break;
        case CallOp::ATAN2: OC_GEN_FUNC_NAME(atan2); break;
        case CallOp::COPYSIGN: OC_GEN_FUNC_NAME(copysign); break;
        case CallOp::COS: OC_GEN_FUNC_NAME(cos); break;
        case CallOp::SIN: OC_GEN_FUNC_NAME(sin); break;
        case CallOp::TAN: OC_GEN_FUNC_NAME(tan); break;
        case CallOp::SINH: OC_GEN_FUNC_NAME(sinh); break;
        case CallOp::COSH: OC_GEN_FUNC_NAME(cosh); break;
        case CallOp::TANH: OC_GEN_FUNC_NAME(tanh); break;
        case CallOp::EXP: OC_GEN_FUNC_NAME(exp); break;
        case CallOp::EXP2: OC_GEN_FUNC_NAME(exp2); break;
        case CallOp::EXP10: OC_GEN_FUNC_NAME(exp10); break;
        case CallOp::LOG: OC_GEN_FUNC_NAME(log); break;
        case CallOp::LOG2: OC_GEN_FUNC_NAME(log2); break;
        case CallOp::LOG10: OC_GEN_FUNC_NAME(log10); break;
        case CallOp::POW: OC_GEN_FUNC_NAME(pow); break;
        case CallOp::SQRT: OC_GEN_FUNC_NAME(sqrt); break;
        case CallOp::RSQRT: OC_GEN_FUNC_NAME(rsqrt); break;
        case CallOp::CEIL: OC_GEN_FUNC_NAME(ceil); break;
        case CallOp::FLOOR: OC_GEN_FUNC_NAME(floor); break;
        case CallOp::ROUND: OC_GEN_FUNC_NAME(round); break;
        case CallOp::FMA: OC_GEN_FUNC_NAME(fma); break;
        case CallOp::CROSS: OC_GEN_FUNC_NAME(cross); break;
        case CallOp::DOT: OC_GEN_FUNC_NAME(dot); break;
        case CallOp::LENGTH: OC_GEN_FUNC_NAME(length); break;
        case CallOp::LENGTH_SQUARED: OC_GEN_FUNC_NAME(length_squared); break;
        case CallOp::DISTANCE: OC_GEN_FUNC_NAME(distance); break;
        case CallOp::DISTANCE_SQUARED: OC_GEN_FUNC_NAME(distance_squared); break;
        case CallOp::NORMALIZE: OC_GEN_FUNC_NAME(normalize); break;
        case CallOp::FACE_FORWARD: OC_GEN_FUNC_NAME(face_forward); break;
        case CallOp::COORDINATE_SYSTEM: OC_GEN_FUNC_NAME(coordinate_system); break;
        case CallOp::DETERMINANT: OC_GEN_FUNC_NAME(det); break;
        case CallOp::TRANSPOSE: OC_GEN_FUNC_NAME(transpose); break;
        case CallOp::INVERSE: OC_GEN_FUNC_NAME(inverse); break;
        case CallOp::SQR: OC_GEN_FUNC_NAME(sqr); break;
        case CallOp::RCP: OC_GEN_FUNC_NAME(rcp); break;
        case CallOp::SIGN: OC_GEN_FUNC_NAME(sign); break;
        case CallOp::FRACT: OC_GEN_FUNC_NAME(fract); break;
        case CallOp::DEGREES: OC_GEN_FUNC_NAME(degrees); break;
        case CallOp::RADIANS: OC_GEN_FUNC_NAME(radians); break;
        case CallOp::SATURATE: OC_GEN_FUNC_NAME(saturate); break;
        case CallOp::SYNCHRONIZE_BLOCK: OC_ASSERT(0); break;
        case CallOp::MAKE_BOOL2: OC_GEN_FUNC_NAME(make_bool2); break;
        case CallOp::MAKE_BOOL3: OC_GEN_FUNC_NAME(make_bool3); break;
        case CallOp::MAKE_BOOL4: OC_GEN_FUNC_NAME(make_bool4); break;
        case CallOp::MAKE_INT2: OC_GEN_FUNC_NAME(make_int2); break;
        case CallOp::MAKE_INT3: OC_GEN_FUNC_NAME(make_int3); break;
        case CallOp::MAKE_INT4: OC_GEN_FUNC_NAME(make_int4); break;
        case CallOp::MAKE_UINT2: OC_GEN_FUNC_NAME(make_uint2); break;
        case CallOp::MAKE_UINT3: OC_GEN_FUNC_NAME(make_uint3); break;
        case CallOp::MAKE_UINT4: OC_GEN_FUNC_NAME(make_uint4); break;
        case CallOp::MAKE_UCHAR2: OC_GEN_FUNC_NAME(make_uchar2); break;
        case CallOp::MAKE_UCHAR3: OC_GEN_FUNC_NAME(make_uchar3); break;
        case CallOp::MAKE_UCHAR4: OC_GEN_FUNC_NAME(make_uchar4); break;
        case CallOp::MAKE_FLOAT2: OC_GEN_FUNC_NAME(make_float2); break;
        case CallOp::MAKE_FLOAT3: OC_GEN_FUNC_NAME(make_float3); break;
        case CallOp::MAKE_FLOAT4: OC_GEN_FUNC_NAME(make_float4); break;
        case CallOp::MAKE_FLOAT2X2: OC_GEN_FUNC_NAME(make_float2x2); break;
        case CallOp::MAKE_FLOAT3X3: OC_GEN_FUNC_NAME(make_float3x3); break;
        case CallOp::MAKE_FLOAT4X4: OC_GEN_FUNC_NAME(make_float4x4); break;
        case CallOp::ATOMIC_EXCH: OC_GEN_FUNC_NAME(atomicExch); break;
        case CallOp::ATOMIC_ADD: OC_GEN_FUNC_NAME(atomicAdd); break;
        case CallOp::ATOMIC_SUB: OC_GEN_FUNC_NAME(atomicSub); break;
        case CallOp::BINDLESS_ARRAY_BUFFER_WRITE: OC_GEN_FUNC_NAME(bindless_array_buffer_write); break;
        case CallOp::BINDLESS_ARRAY_BYTE_BUFFER_WRITE: OC_GEN_FUNC_NAME(bindless_array_byte_buffer_write); break;
        case CallOp::BINDLESS_ARRAY_BUFFER_SIZE: OC_GEN_FUNC_NAME(bindless_array_buffer_size); break;
        case CallOp::BINDLESS_ARRAY_BYTE_BUFFER_READ: {
            current_scratch() << "oc_bindless_array_byte_buffer_read<";
            _emit_type_name(expr->type());
            current_scratch() << ">";
            break;
        }
        case CallOp::BINDLESS_ARRAY_BUFFER_READ: {
            current_scratch() << "oc_bindless_array_buffer_read<";
            _emit_type_name(expr->type());
            current_scratch() << ">";
            break;
        }
        case CallOp::BYTE_BUFFER_WRITE: OC_GEN_FUNC_NAME(byte_buffer_write); break;
        case CallOp::BYTE_BUFFER_READ: {
            auto t_args = expr->template_args();
            ocarina::visit(
                [&]<typename T>(T &&t) {
                    if constexpr (is_integral_v<T>) {
                        current_scratch() << "oc_byte_buffer_read<" << int(t) << ">";
                    } else {
                        current_scratch() << "oc_byte_buffer_read<";
                        _emit_type_name(t);
                        current_scratch() << ">";
                    }
                },
                t_args[0]);
            break;
        }
        case CallOp::BINDLESS_ARRAY_TEX_SAMPLE: {
            auto t_args = expr->template_args();
            uint N = std::get<uint>(t_args[0]);
            current_scratch() << "oc_bindless_array_tex_sample<" << N << ">";
            break;
        }
        case CallOp::UNREACHABLE: current_scratch() << "__builtin_unreachable"; break;
        case CallOp::MAKE_RAY: OC_GEN_FUNC_NAME(make_ray); break;
        case CallOp::TRACE_ANY: OC_GEN_FUNC_NAME(trace_any); break;
        case CallOp::RAY_OFFSET_ORIGIN: OC_GEN_FUNC_NAME(offset_ray_origin); break;
        case CallOp::TRACE_CLOSEST: OC_GEN_FUNC_NAME(trace_closest); break;
        case CallOp::IS_NULL_BUFFER: OC_GEN_FUNC_NAME(is_null_buffer); break;
        case CallOp::IS_NULL_TEXTURE: OC_GEN_FUNC_NAME(is_null_texture); break;
        case CallOp::BUFFER_SIZE: OC_GEN_FUNC_NAME(buffer_size); break;
        case CallOp::BYTE_BUFFER_SIZE: OC_GEN_FUNC_NAME(buffer_size); break;
        case CallOp::TEX_SAMPLE: {
            auto t_args = expr->template_args();
            uint N = std::get<uint>(t_args[0]);
            current_scratch() << "oc_tex_sample_float<" << N << ">";
            break;
        }
        case CallOp::TEX_READ: {
            auto t_args = expr->template_args();
            auto output_type = t_args[0];
            current_scratch() << ocarina::format("oc_texture_read<oc_{}>",
                                                 std::get<const Type *>(output_type)->name());
            break;
        }
        case CallOp::TEX_WRITE: {
            current_scratch() << "oc_texture_write";
            break;
        }
        case CallOp::COUNT: break;
        default: OC_ASSERT(0); break;
    }
#undef OC_GEN_FUNC_NAME
    emit_act_arguments(expr);
}

void CUDACodegen::visit(const MemberExpr *expr) noexcept {
    if (expr->is_swizzle()) {
        static constexpr std::string_view xyzw[] = {"x", "y", "z", "w"};
        int swizzle_size = expr->swizzle_size();
        if (swizzle_size == 1) {
            expr->parent()->accept(*this);
            current_scratch() << ".";
            current_scratch() << xyzw[expr->swizzle_index(0)];
        } else {
            _emit_type_name(expr->type());
            current_scratch() << "(";
            for (int i = 0; i < swizzle_size; ++i) {
                expr->parent()->accept(*this);
                current_scratch() << ".";
                current_scratch() << xyzw[expr->swizzle_index(i)] << ",";
            }
            current_scratch().pop_back();
            current_scratch() << ")";
        }
    } else {
        expr->parent()->accept(*this);
        current_scratch() << ".";
        _emit_member_name(expr->parent()->type(), expr->member_index());
    }
}

void CUDACodegen::_emit_raytracing_param(const Function &f) noexcept {
    current_scratch() << "struct Params {";
    _emit_newline();
    indent_inc();
    size_t offset = 0;
    ocarina::vector<MemoryBlock> blocks;
    blocks.reserve(f.arguments().size() + f.captured_resources().size());
    auto func = [&](const Variable &arg) {
        _emit_indent();
        size_t size = CUDADevice::size(arg.type());
        size_t alignment = CUDADevice::alignment(arg.type());
        blocks.emplace_back(nullptr, size, alignment, CUDADevice::max_member_size(arg.type()));
        offset = mem_offset(offset, alignment);
        current_scratch() << ocarina::format("/* {} bytes */", size);
        _emit_newline();
        _emit_indent();
        _emit_variable_define(arg);
        offset += size;
        current_scratch() << ";";
        _emit_newline();
    };

    for (const Variable &arg : f.arguments()) {
        func(arg);
    }

    f.for_each_captured_resource([&](const CapturedResource &uniform) {
        const Variable &arg = uniform.expression()->variable();
        func(arg);
    });
    indent_dec();
    current_scratch() << "};";
    _emit_newline();
    current_scratch() << ocarina::format("static_assert(sizeof(Params) == {});", structure_size(blocks));
    _emit_newline();
    current_scratch() << "extern \"C\" __constant__ Params params;";
    _emit_newline();
}

void CUDACodegen::_emit_function(const Function &f) noexcept {
    if (has_generated(&f)) {
        return;
    }
    switch (f.tag()) {
        case Function::Tag::KERNEL: current_scratch() << "extern \"C\" __global__ "; break;
        case Function::Tag::CALLABLE: current_scratch() << "__device__ "; break;
    }
    CppCodegen::_emit_function(f);
}

void CUDACodegen::_emit_builtin_vars_define(const Function &f) noexcept {
    const char *str = "oc_uint3 d_idx = oc_make_uint3(blockIdx.x * blockDim.x + threadIdx.x,"
                      "blockIdx.y * blockDim.y + threadIdx.y,"
                      "blockIdx.z * blockDim.z + threadIdx.z);";
    if (f.is_general_kernel()) {
        _emit_indent();
        current_scratch() << str;
        _emit_newline();
        _emit_indent();
        current_scratch() << "if (oc_any(d_idx >= d_dim)) { return; }";
        _emit_newline();
    } else if (f.is_raytracing_kernel()) {
        _emit_indent();
        current_scratch() << "oc_uint3 d_idx = getLaunchIndex();";
        _emit_newline();
        _emit_indent();
        current_scratch() << "oc_uint3 d_dim = getLaunchDim();";
        _emit_newline();

        auto emit_arg = [&](const Variable &arg) {
            _emit_indent();
            current_scratch() << "auto ";
            _emit_variable_name(arg);
            current_scratch() << " = params.";
            _emit_variable_name(arg);
            current_scratch() << ";";
            _emit_newline();
        };

        for (const Variable &arg : f.arguments()) {
            emit_arg(arg);
        }

        f.for_each_captured_resource([&](const CapturedResource &uniform) {
            emit_arg(uniform.expression()->variable());
        });
    } else if (f.is_callable()) {
    }
    CppCodegen::_emit_builtin_vars_define(f);
}

void CUDACodegen::_emit_builtin_var(Variable v) noexcept {
    _emit_type_name(v.type());
    _emit_space();
    _emit_variable_name(v);
    using Tag = Variable::Tag;
    current_scratch() << " = ";
    switch (v.tag()) {
        case Tag::BLOCK_IDX:
            current_scratch() << "oc_make_uint3(blockIdx.x, blockIdx.y, blockIdx.z)";
            break;
        case Tag::THREAD_IDX:
            current_scratch() << "oc_make_uint3(threadIdx.x, threadIdx.y, threadIdx.z)";
            break;
        case Tag::THREAD_ID:
            current_scratch() << "(blockIdx.x + blockIdx.y * gridDim.x"
                                 "+ gridDim.x * gridDim.y * blockIdx.z) * "
                                 "(blockDim.x * blockDim.y * blockDim.z)"
                                 " + (threadIdx.z * (blockDim.x * blockDim.y))"
                                 " + (threadIdx.y * blockDim.x) + threadIdx.x";
            break;
        case Tag::DISPATCH_IDX:
            current_scratch() << "d_idx";
            break;
        case Tag::DISPATCH_ID:
            current_scratch() << "d_idx.z * d_dim.x * d_dim.y + d_dim.x * d_idx.y + d_idx.x";
            break;
        case Tag::DISPATCH_DIM: {
            current_scratch() << "d_dim";
            break;
        }
        default:
            OC_ASSERT(0);
            break;
    }
}

void CUDACodegen::_emit_arguments(const Function &f) noexcept {
    current_scratch() << "(";
    if (f.is_general_kernel()) {
        for (const auto &v : f.all_arguments()) {
            _emit_argument(v);
        }
        Variable dispatch_dim = const_cast<Function &>(f).create_variable(Type::of<uint3>(), Variable::Tag::LOCAL, "d_dim");
        _emit_variable_define(dispatch_dim);
    } else if (f.is_callable()) {
        for (const auto &v : f.all_arguments()) {
            _emit_argument(v);
        }
        Variable dispatch_dim = const_cast<Function &>(f).create_variable(Type::of<uint3>(), Variable::Tag::LOCAL, "d_dim");
        _emit_argument(dispatch_dim);
        Variable dispatch_idx = const_cast<Function &>(f).create_variable(Type::of<uint3>(), Variable::Tag::LOCAL, "d_idx");
        _emit_variable_define(dispatch_idx);
    }
    current_scratch() << ")";
}

void CUDACodegen::_emit_type_name(const Type *type) noexcept {
    if (type == nullptr) {
        current_scratch() << "void";
    } else {
        switch (type->tag()) {
            case Type::Tag::BOOL:
            case Type::Tag::FLOAT:
            case Type::Tag::INT:
            case Type::Tag::UINT:
            case Type::Tag::UCHAR:
            case Type::Tag::CHAR:
            case Type::Tag::USHORT:
            case Type::Tag::SHORT:
            case Type::Tag::VECTOR:
            case Type::Tag::MATRIX:
            case Type::Tag::UINT64T:
                current_scratch() << TYPE_PREFIX << type->name();
                break;
            case Type::Tag::ARRAY:
                current_scratch() << TYPE_PREFIX "array<";
                _emit_type_name(type->element());
                current_scratch() << "," << type->dimension() << ">";
                break;
            case Type::Tag::STRUCTURE:
                _emit_struct_name(type);
                break;
            case Type::Tag::BUFFER:
                current_scratch() << "OCBuffer<";
                _emit_type_name(type->element());
                current_scratch() << ">";
                break;
            case Type::Tag::BYTE_BUFFER:
                current_scratch() << "OCBuffer<oc_uchar>";
                break;
            case Type::Tag::TEXTURE:
                current_scratch() << "OCTexture";
                break;
            case Type::Tag::BINDLESS_ARRAY:
                current_scratch() << "OCBindlessArray";
                break;
            case Type::Tag::ACCEL:
                current_scratch() << "OptixTraversableHandle";
                break;
            case Type::Tag::NONE:
                break;
        }
    }
}

void CUDACodegen::_emit_struct_name(const Type *type) noexcept {
    OC_ERROR_IF(type->cname().empty());
    if (type->is_builtin_struct()) {
        current_scratch() << type->simple_cname();
    } else {
        Codegen::_emit_struct_name(type);
    }
}

}// namespace ocarina