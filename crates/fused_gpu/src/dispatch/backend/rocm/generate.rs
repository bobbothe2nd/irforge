use log::debug;

use crate::{
    dispatch::{
        CompilationOptions, DebugCompilationOptions, TargetFlags,
        backend::{
            DType, Op, Param, ParamTy, SharedAlloc, SimpleDType, ValueId,
            codegen::{self, def_var_c, newline},
            kernel::RawKernel,
        },
    }, errors::Error,
};

use core::fmt::Write;

pub fn generate_hip(
    src: &RawKernel,
    params: &[Param],
    options: &CompilationOptions,
) -> Result<String, Error> {
    let Ok(mut out) = "
#include <cstdint>
#include <bit>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

using namespace hip;

"
    .parse::<String>();

    if options.target.flags.contains(TargetFlags::LIN_ACC) {
        out.push_str(
            "
#include <hip/hip_wmma.h>

using namespace hip::wmma;

",
        );
    }

    let pretty_print = options
        .debug
        .contains(DebugCompilationOptions::PRETTY_PRINT_IR);

    out.push_str("extern \"C\" __global__\nvoid kernel(");

    gen_args(&mut out, params);

    out.push_str(") {");

    gen_shared(&mut out, &src.shared, pretty_print);

    let mut nesting = 1;

    for op in &src.ops {
        newline(pretty_print, &mut out, nesting);

        process_op(&mut out, op, &mut nesting, options, src)?;
    }

    newline(pretty_print, &mut out, 0);

    out.push('}');

    if pretty_print {
        debug!("generated kernel {}:\n{out}", src.name);
    }

    Ok(out)
}

fn gen_shared(out: &mut String, shared: &[SharedAlloc], pretty_print: bool) {
    for (i, s) in shared.iter().enumerate() {
        newline(pretty_print, out, 0);

        let _ = write!(out, "__shared__ {} shared{i}[{}];", s.dtype.fmt_c(), s.size);
    }
}

fn gen_args(out: &mut String, params: &[Param]) {
    for (i, param) in params.iter().enumerate() {
        if i != 0 {
            let _ = out.write_str(", ");
        }

        let read_only = if param.ty == ParamTy::ReadWrite {
            ""
        } else {
            "const "
        };

        let _ = write!(
            out,
            "{}{}* __restrict__ param{}",
            read_only,
            param.dtype.fmt_c(),
            param.pid
        );
    }
}

fn process_op(
    out: &mut String,
    op: &Op,
    nesting: &mut usize,
    options: &CompilationOptions,
    kernel: &RawKernel,
) -> Result<(), Error> {
    match op {
        Op::Nop => {}

        Op::DefineVar { id } => def_var_c(out, nesting, *id, kernel, options, process_op)?,

        Op::OverwriteVar { id, val } => {
            let _ = write!(out, "v{id} = {};", render_val(*val, kernel, options)?);
        }

        Op::AddAssign { id, val } => {
            let _ = write!(out, "v{id} += {};", render_val(*val, kernel, options)?);
        }

        Op::MulAssign { id, val } => {
            let _ = write!(out, "v{id} *= {};", render_val(*val, kernel, options)?);
        }

        Op::DivAssign { id, val } => {
            let _ = write!(out, "v{id} /= {};", render_val(*val, kernel, options)?);
        }

        Op::SubAssign { id, val } => {
            let _ = write!(out, "v{id} -= {};", render_val(*val, kernel, options)?);
        }

        Op::ShlAssign { id, val } => {
            let _ = write!(out, "v{id} <<= {};", render_val(*val, kernel, options)?);
        }

        Op::ShrAssign { id, val } => {
            let _ = write!(out, "v{id} >>= {};", render_val(*val, kernel, options)?);
        }

        Op::CopyVar { id } => {
            let _ = out.write_str(&render_val(*id, kernel, options)?);
        }

        Op::ConstF64 { val } => {
            let _ = write!(out, "{val:?}");
        }

        Op::ConstF32 { val } => {
            let _ = write!(out, "{val:?}f");
        }

        Op::ConstF16 { val } => {
            let _ = write!(out, "__float2half({val:?}f)");
        }

        Op::ConstBf16 { val } => {
            let _ = write!(out, "__float2bfloat16({val:?}f)");
        }

        Op::ConstU32 { val } => {
            let _ = write!(out, "{val}u");
        }

        Op::ConstI32 { val } => {
            let _ = write!(out, "{val}");
        }

        Op::ReadMeta { param, field } => {
            let _ = write!(out, "param{param}[{field}]");
        }

        Op::Eq { a, b } => {
            let _ = write!(
                out,
                "({}) == ({})",
                render_val(*a, kernel, options)?,
                render_val(*b, kernel, options)?
            );
        }

        Op::Ne { a, b } => {
            let _ = write!(
                out,
                "({}) != ({})",
                render_val(*a, kernel, options)?,
                render_val(*b, kernel, options)?
            );
        }

        Op::Lt { a, b } => {
            let _ = write!(
                out,
                "({}) < ({})",
                render_val(*a, kernel, options)?,
                render_val(*b, kernel, options)?
            );
        }

        Op::Gt { a, b } => {
            let _ = write!(
                out,
                "({}) > ({})",
                render_val(*a, kernel, options)?,
                render_val(*b, kernel, options)?
            );
        }

        Op::Le { a, b } => {
            let _ = write!(
                out,
                "({}) <= ({})",
                render_val(*a, kernel, options)?,
                render_val(*b, kernel, options)?
            );
        }

        Op::Ge { a, b } => {
            let _ = write!(
                out,
                "({}) >= ({})",
                render_val(*a, kernel, options)?,
                render_val(*b, kernel, options)?
            );
        }

        Op::LocalId { axis } => {
            let _ = write!(out, "threadIdx.{axis}");
        }

        Op::BlockId { axis } => {
            let _ = write!(out, "blockIdx.{axis}");
        }

        Op::GlobalId { axis } => {
            let _ = write!(
                out,
                "blockIdx.{axis} * {} + threadIdx.{axis}",
                kernel.block[*axis]
            );
        }

        Op::Add { a, b } => {
            let _ = write!(
                out,
                "({}) + ({})",
                render_val(*a, kernel, options)?,
                render_val(*b, kernel, options)?
            );
        }

        Op::Sub { a, b } => {
            let _ = write!(
                out,
                "({}) - ({})",
                render_val(*a, kernel, options)?,
                render_val(*b, kernel, options)?
            );
        }

        Op::Mul { a, b } => {
            let _ = write!(
                out,
                "({}) * ({})",
                render_val(*a, kernel, options)?,
                render_val(*b, kernel, options)?
            );
        }

        Op::Div { a, b } => {
            let _ = write!(
                out,
                "({}) / ({})",
                render_val(*a, kernel, options)?,
                render_val(*b, kernel, options)?
            );
        }

        Op::Mod { a, b } => {
            let _ = write!(
                out,
                "({}) % ({})",
                render_val(*a, kernel, options)?,
                render_val(*b, kernel, options)?
            );
        }

        Op::Pow { a, b } => {
            let _ = write!(
                out,
                "pow({}, {})",
                render_val(*a, kernel, options)?,
                render_val(*b, kernel, options)?
            );
        }

        Op::Shl { a, b } => {
            let _ = write!(
                out,
                "({}) << ({})",
                render_val(*a, kernel, options)?,
                render_val(*b, kernel, options)?
            );
        }

        Op::Shr { a, b } => {
            let _ = write!(
                out,
                "({}) >> ({})",
                render_val(*a, kernel, options)?,
                render_val(*b, kernel, options)?
            );
        }

        Op::Fma { a, b, c } => {
            let symbol = if kernel.vals[*a].dtype == DType::Simple(SimpleDType::F32) {
                "fmaf"
            } else {
                "fma"
            };
            let _ = write!(
                out,
                "{symbol}({}, {}, {})",
                render_val(*a, kernel, options)?,
                render_val(*b, kernel, options)?,
                render_val(*c, kernel, options)?
            );
        }

        Op::Max { a, b } => {
            let _ = write!(
                out,
                "max({}, {})",
                render_val(*a, kernel, options)?,
                render_val(*b, kernel, options)?
            );
        }

        Op::Min { a, b } => {
            let _ = write!(
                out,
                "min({}, {})",
                render_val(*a, kernel, options)?,
                render_val(*b, kernel, options)?
            );
        }

        Op::Select { cond, a, b } => {
            let _ = write!(
                out,
                "({} ? {} : {})",
                render_val(*cond, kernel, options)?,
                render_val(*a, kernel, options)?,
                render_val(*b, kernel, options)?
            );
        }

        Op::Exp { x } => {
            let _ = write!(out, "exp({})", render_val(*x, kernel, options)?);
        }

        Op::Abs { x } => {
            let _ = write!(out, "abs({})", render_val(*x, kernel, options)?);
        }

        Op::Neg { x } => {
            let _ = write!(out, "-({})", render_val(*x, kernel, options)?);
        }

        Op::Log { x } => {
            let _ = write!(out, "log({})", render_val(*x, kernel, options)?);
        }

        Op::Tanh { x } => {
            let _ = write!(out, "tanh({})", render_val(*x, kernel, options)?);
        }

        Op::Sqrt { x } => {
            let _ = write!(out, "sqrt({})", render_val(*x, kernel, options)?);
        }

        Op::ParamLoad { param, index } => {
            let _ = write!(out, "param{param}[{}]", render_val(*index, kernel, options)?);
        }

        Op::Not { cond } => {
            let _ = write!(out, "!({})", render_val(*cond, kernel, options)?);
        }

        Op::Cast { id, dtype } => {
            let _ = write!(
                out,
                "static_cast<{}>({})",
                dtype.fmt_c(),
                render_val(*id, kernel, options)?
            );
        }

        Op::BitCast { id, dtype } => {
            let _ = write!(
                out,
                "__builtin_bit_cast({}, {})",
                dtype.fmt_c(),
                render_val(*id, kernel, options)?
            );
        }

        Op::ParamStore {
            param,
            index,
            val,
        } => {
            let _ = write!(
                out,
                "param{param}[{}] = {};",
                render_val(*index, kernel, options)?,
                render_val(*val, kernel, options)?
            );
        }

        Op::ParamAccum {
            param,
            index,
            val,
        } => {
            let _ = write!(
                out,
                "param{param}[{}] += {};",
                render_val(*index, kernel, options)?,
                render_val(*val, kernel, options)?
            );
        }

        Op::ParamMul {
            param,
            index,
            val,
        } => {
            let _ = write!(
                out,
                "param{param}[{}] *= {};",
                render_val(*index, kernel, options)?,
                render_val(*val, kernel, options)?
            );
        }

        Op::ParamDiv {
            param,
            index,
            val,
        } => {
            let _ = write!(
                out,
                "param{param}[{}] /= {};",
                render_val(*index, kernel, options)?,
                render_val(*val, kernel, options)?
            );
        }

        Op::ParamSub {
            param,
            index,
            val,
        } => {
            let _ = write!(
                out,
                "param{param}[{}] -= {};",
                render_val(*index, kernel, options)?,
                render_val(*val, kernel, options)?
            );
        }

        Op::ParamShl {
            param,
            index,
            val,
        } => {
            let _ = write!(
                out,
                "param{param}[{}] <<= {};",
                render_val(*index, kernel, options)?,
                render_val(*val, kernel, options)?
            );
        }

        Op::ParamShr {
            param,
            index,
            val,
        } => {
            let _ = write!(
                out,
                "param{param}[{}] >>= {};",
                render_val(*index, kernel, options)?,
                render_val(*val, kernel, options)?
            );
        }

        Op::SharedLoad { mem, index } => {
            let _ = write!(out, "shared{mem}[{}]", render_val(*index, kernel, options)?);
        }

        Op::SharedStore { mem, index, val } => {
            let _ = write!(
                out,
                "shared{mem}[{}] = {};",
                render_val(*index, kernel, options)?,
                render_val(*val, kernel, options)?
            );
        }

        Op::SharedAccum { mem, index, val } => {
            let _ = write!(
                out,
                "shared{mem}[{}] += {};",
                render_val(*index, kernel, options)?,
                render_val(*val, kernel, options)?
            );
        }

        Op::SharedMul { mem, index, val } => {
            let _ = write!(
                out,
                "shared{mem}[{}] *= {};",
                render_val(*index, kernel, options)?,
                render_val(*val, kernel, options)?
            );
        }

        Op::SharedDiv { mem, index, val } => {
            let _ = write!(
                out,
                "shared{mem}[{}] /= {};",
                render_val(*index, kernel, options)?,
                render_val(*val, kernel, options)?
            );
        }

        Op::SharedSub { mem, index, val } => {
            let _ = write!(
                out,
                "shared{mem}[{}] -= {};",
                render_val(*index, kernel, options)?,
                render_val(*val, kernel, options)?
            );
        }

        Op::SharedShl { mem, index, val } => {
            let _ = write!(
                out,
                "shared{mem}[{}] <<= {};",
                render_val(*index, kernel, options)?,
                render_val(*val, kernel, options)?
            );
        }

        Op::SharedShr { mem, index, val } => {
            let _ = write!(
                out,
                "shared{mem}[{}] >>= {};",
                render_val(*index, kernel, options)?,
                render_val(*val, kernel, options)?
            );
        }

        Op::ForLoopBegin { index, end, step } => {
            let _ = write!(
                out,
                "for (; v{index} < {}; v{index} += {}) {{",
                render_val(*end, kernel, options)?,
                render_val(*step, kernel, options)?,
            );
            *nesting += 1;
        }

        Op::ForeverLoopBegin => {
            let _ = out.write_str("for (;;) {{");
            *nesting += 1;
        }

        Op::IfBegin { cond } => {
            let _ = write!(out, "if ({}) {{", render_val(*cond, kernel, options)?);
            *nesting += 1;
        }

        Op::ElseBegin => {
            let _ = out.write_str("else {{");
            *nesting += 1;
        }

        Op::StartScope => {
            let _ = out.write_str("{{");
            *nesting += 1;
        }

        Op::EndScope => {
            let _ = out.write_str("}}");
            *nesting -= 1;
        }

        Op::Barrier => {
            let _ = out.write_str("__syncthreads();");
        }

        Op::Return => {
            let _ = out.write_str("return;");
        }

        Op::Continue => {
            let _ = out.write_str("continue;");
        }

        Op::Break => {
            let _ = out.write_str("break;");
        }

        Op::FillFragment { frag, val } => {
            let _ = write!(
                out,
                "fill_fragment({}, {});",
                render_val(*frag, kernel, options)?,
                render_val(*val, kernel, options)?
            );
        }

        Op::Index { pointer, index } => {
            let _ = write!(
                out,
                "{}[{}]",
                render_val(*pointer, kernel, options)?,
                render_val(*index, kernel, options)?,
            );
        }

        Op::LoadMatrixSync {
            frag,
            param,
            index,
            stride,
        } => {
            let _ = write!(
                out,
                "load_matrix_sync({}, param{param} + {}, {});",
                render_val(*frag, kernel, options)?,
                render_val(*index, kernel, options)?,
                render_val(*stride, kernel, options)?
            );
        }

        Op::MmaSync { acc, a, b, c } => {
            let _ = write!(
                out,
                "mma_sync({}, {}, {}, {});",
                render_val(*acc, kernel, options)?,
                render_val(*a, kernel, options)?,
                render_val(*b, kernel, options)?,
                render_val(*c, kernel, options)?
            );
        }

        Op::StoreMatrixSync {
            param,
            index,
            frag,
            stride,
            layout,
        } => {
            let _ = write!(
                out,
                "store_matrix_sync(param{param} +{}, {}, {}, mem_{});",
                render_val(*index, kernel, options)?,
                render_val(*frag, kernel, options)?,
                render_val(*stride, kernel, options)?,
                layout.fmt_wmma(),
            );
        }
    }

    Ok(())
}

fn render_val(id: ValueId, kernel: &RawKernel, options: &CompilationOptions) -> Result<String, Error> {
    codegen::render_val(id, kernel, options, process_op)
}
