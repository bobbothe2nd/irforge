use crate::{
    dispatch::{CompilationOptions, DebugCompilationOptions, backend::{
        Axis, DType, Op, Param, ParamTy, SimpleDType, TensorLayout, ValueId, ValueState, codegen::{self, def_var_wgsl, newline, tab}, kernel::RawKernel,
    }}, errors::{Error, ErrorKind},
};
use core::fmt::Write;
use std::{string::String, vec::Vec};

use log::debug;
pub use wgpu::{BindGroupLayoutEntry, BindingType, BufferBindingType, ShaderStages};

#[inline]
pub(super) fn generate_layout_desc(params: &[Param]) -> Vec<BindGroupLayoutEntry> {
    params
        .iter()
        .map(|x| {
            let ty = match x.ty {
                ParamTy::Uniform => BufferBindingType::Uniform,
                ParamTy::ReadOnly => BufferBindingType::Storage { read_only: true },
                ParamTy::ReadWrite => BufferBindingType::Storage { read_only: false },
            };

            BindGroupLayoutEntry {
                binding: x.pid as u32,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        })
        .collect::<Vec<_>>()
}

#[inline]
pub fn generate_wgsl(
    kernel: &RawKernel,
    params: &[Param],
    options: &CompilationOptions,
) -> Result<String, Error> {
    let pretty_print = options
        .debug
        .contains(DebugCompilationOptions::PRETTY_PRINT_IR);

    let mut out = "enable f16;".parse().map_err(|_| Error {
        msg: "failed to copy &'static str to String",
        kind: ErrorKind::InternalError,
        ctx: (),
    })?;

    newline(pretty_print, &mut out, 0);

    emit_bindings(kernel, params, &mut out, pretty_print)?;
    newline(pretty_print, &mut out, 0);

    emit_entry(kernel, &mut out, options, pretty_print)?;

    if pretty_print {
        debug!("generated kernel:\n{out}");
    }

    Ok(out)
}

#[inline]
fn emit_bindings(
    kernel: &RawKernel,
    params: &[Param],
    out: &mut String,
    pretty_print: bool,
) -> Result<(), Error> {
    let _ = write!(out, "struct Meta {{");
    newline(pretty_print, out, 0);
    let _ = write!(out, "f0: f32,");
    for f in 1..kernel.meta.fields {
        newline(pretty_print, out, 1);
        let _ = write!(out, "f{f}: u32,");
    }
    newline(pretty_print, out, 0);
    let _ = write!(out, "}}");

    newline(pretty_print, out, 0);

    for p in params {
        let var_type = match p.ty {
            ParamTy::ReadOnly => "var<storage, read>",
            ParamTy::ReadWrite => "var<storage, read_write>",
            ParamTy::Uniform => "var<uniform>",
        };

        newline(pretty_print, out, 0);

        let pid = p.pid;

        if p.ty == ParamTy::Uniform && p.dtype == SimpleDType::U32 {
            let _ = write!(
                out,
                "@group(0) @binding({pid}) {var_type} param{pid}: Meta;"
            );
        } else {
            let _ = write!(
                out,
                "@group(0) @binding({pid}) {var_type} param{pid}: array<{}>;",
                p.dtype.fmt_wgsl()?,
            );
        }
    }

    newline(pretty_print, out, 0);

    for (i, s) in kernel.shared.iter().enumerate() {
        newline(pretty_print, out, 0);
        let _ = write!(
            out,
            "var<workgroup> shared{i}: array<{}, ({})>;",
            s.dtype.fmt_wgsl()?,
            s.size
        );
    }

    Ok(())
}

#[inline]
fn emit_entry(kernel: &RawKernel, out: &mut String, options: &CompilationOptions, pretty_print: bool) -> Result<(), Error> {
    newline(pretty_print, out, 0);
    let _ = write!(
        out,
        "@compute @workgroup_size({}, ({}), ({})) ",
        kernel.block[Axis::X], kernel.block[Axis::Y], kernel.block[Axis::Z]
    );

    newline(pretty_print, out, 0);
    let _ = write!(out, "fn main(");

    newline(pretty_print, out, 1);
    let _ = write!(out, "@builtin(local_invocation_id) lid: vec3<u32>,");

    newline(pretty_print, out, 1);
    let _ = write!(out, "@builtin(workgroup_id) bid: vec3<u32>,");

    newline(pretty_print, out, 1);
    let _ = write!(out, "@builtin(global_invocation_id) gid: vec3<u32>,");

    newline(pretty_print, out, 0);
    let _ = write!(out, ") {{");

    emit_ops(kernel, out, options, pretty_print)?;
    newline(pretty_print, out, 0);

    let _ = write!(out, "}}");

    Ok(())
}

#[inline]
fn emit_ops(kernel: &RawKernel, out: &mut String, options: &CompilationOptions, pretty_print: bool) -> Result<(), Error> {
    let mut nesting = 0;

    for op in &kernel.ops {
        if let Op::DefineVar { id } = op
            && matches!(
                kernel.vals[*id].state,
                ValueState::Inline | ValueState::Masked
            )
        {
            continue;
        }

        newline(pretty_print, out, nesting);

        if *op != Op::EndScope {
            tab(pretty_print, out);
        }

        process_op(out, op, &mut nesting, options, kernel)?;
    }

    Ok(())
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

        Op::DefineVar { id } => def_var_wgsl(out, nesting, *id, kernel, options, process_op)?,

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
            let _ = write!(out, "{val}d");
        }

        Op::ConstF32 { val } => {
            let _ = write!(out, "{val}f");
        }

        Op::ConstF16 { val } => {
            let _ = write!(out, "{val}h");
        }

        Op::ConstBf16 { val: _ } => {
            return Err(Error {
                msg: "cannot evaluate constant `bf16`",
                kind: ErrorKind::InvalidDType,
                ctx: (),
            });
        }

        Op::ConstU32 { val } => {
            let _ = write!(out, "{val}u");
        }

        Op::ConstI32 { val } => {
            let _ = write!(out, "{val}");
        }

        Op::ReadMeta { param, field } => {
            let _ = write!(out, "param{param}.f{field}");
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
            let _ = write!(out, "lid.{axis}");
        }

        Op::BlockId { axis } => {
            let _ = write!(out, "bid.{axis}");
        }

        Op::GlobalId { axis } => {
            let _ = write!(out, "gid.{axis}");
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
            let _ = write!(
                out,
                "fma({}, {}, {})",
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
                "select({}, {}, {})",
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
            let _ = write!(out, "{}({})", dtype.fmt_wgsl()?, render_val(*id, kernel, options)?);
        }

        Op::BitCast { id, dtype } => {
            let _ = write!(
                out,
                "bitcast<{}>({})",
                dtype.fmt_wgsl()?,
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
            let _ = write!(out, "loop {{");
            *nesting += 1;
        }

        Op::IfBegin { cond } => {
            let _ = write!(out, "if ({}) {{", render_val(*cond, kernel, options)?);
            *nesting += 1;
        }

        Op::ElseBegin => {
            let _ = write!(out, "else {{");
            *nesting += 1;
        }

        Op::StartScope => {
            let _ = write!(out, "{{");
        }

        Op::EndScope => {
            let _ = write!(out, "}}");
            *nesting -= 1;
        }

        Op::Barrier => {
            let _ = write!(out, "workgroupBarrier();");
        }

        Op::Return => {
            let _ = write!(out, "return;");
        }

        Op::Continue => {
            let _ = write!(out, "continue;");
        }

        Op::Break => {
            let _ = write!(out, "break;");
        }

        Op::Index { pointer, index } => {
            let _ = write!(out, "{}[{}]", render_val(*pointer, kernel, options)?, render_val(*index, kernel, options)?);
        }

        Op::FillFragment { frag, val } => {
            if !options.debug.contains(DebugCompilationOptions::SOFT_MMA) {
                return Err(Error {
                    msg: "MMA not supported",
                    kind: ErrorKind::UnsupportedFeature,
                    ctx: (),
                });
            }

            let val = render_val(*val, kernel, options)?;

            let _ = write!(
                out,
                "for (var i: u32 = 0u; i < 256u; i++) {{ \
                    v{frag}[i] = {val}; \
                }}"
            );
        }

        Op::LoadMatrixSync {
            frag,
            param,
            index,
            stride,
        } => {
            if !options.debug.contains(DebugCompilationOptions::SOFT_MMA) {
                return Err(Error {
                    msg: "MMA not supported",
                    kind: ErrorKind::UnsupportedFeature,
                    ctx: (),
                });
            }

            let index = render_val(*index, kernel, options)?;
            let stride = render_val(*stride, kernel, options)?;

            let frag_index = render_frag_index(*frag, "r", "c", kernel)?;

            let _ = write!(
                out,
                "for (var r: u32 = 0u; r < 16u; r++) {{ \
                    for (var c: u32 = 0u; c < 16u; c++) {{ \
                        v{frag}[{frag_index}] = \
                            param{param}[({index}) + r * ({stride}) + c]; \
                    }} \
                }}"
            );
        }

        Op::MmaSync { acc, a, b, c } => {
            if !options.debug.contains(DebugCompilationOptions::SOFT_MMA) {
                return Err(Error {
                    msg: "MMA not supported",
                    kind: ErrorKind::UnsupportedFeature,
                    ctx: (),
                });
            }

            let a_mk = render_frag_index(*a, "m", "k", kernel)?;
            let b_kn = render_frag_index(*b, "k", "n", kernel)?;
            let c_mn = render_frag_index(*c, "m", "n", kernel)?;
            let acc_mn = render_frag_index(*acc, "m", "n", kernel)?;

            let _ = write!(
                out,
                "for (var m: u32 = 0u; m < 16u; m++) {{ \
                    for (var n: u32 = 0u; n < 16u; n++) {{ \
                        var sum: f32 = f32(v{c}[{c_mn}]); \
                        for (var k: u32 = 0u; k < 16u; k++) {{ \
                            sum = fma( \
                                f32(v{a}[{a_mk}]), \
                                f32(v{b}[{b_kn}]), \
                                sum \
                            ); \
                        }} \
                        v{acc}[{acc_mn}] = sum; \
                    }} \
                }}"
            );
        }

        Op::StoreMatrixSync {
            param,
            index,
            stride,
            frag,
            layout: _,
        } => {
            if !options.debug.contains(DebugCompilationOptions::SOFT_MMA) {
                return Err(Error {
                    msg: "MMA not supported",
                    kind: ErrorKind::UnsupportedFeature,
                    ctx: (),
                });
            }

            let index = render_val(*index, kernel, options)?;
            let stride = render_val(*stride, kernel, options)?;

            let frag_index = render_frag_index(*frag, "r", "c", kernel)?;

            let _ = write!(
                out,
                "for (var r: u32 = 0u; r < 16u; r++) {{ \
                    for (var c: u32 = 0u; c < 16u; c++) {{ \
                        param{param}[({index}) + r * ({stride}) + c] = \
                            v{frag}[{frag_index}]; \
                    }} \
                }}"
            );
        }
    }

    Ok(())
}

fn render_frag_index(
    frag: ValueId,
    row: &str,
    col: &str,
    kernel: &RawKernel,
) -> Result<String, Error> {
    match kernel.vals[frag].dtype {
        DType::MmaA {
            layout: TensorLayout::RowMajor,
            ..
        }
        | DType::MmaB {
            layout: TensorLayout::RowMajor,
            ..
        }
        | DType::MmaAccum { .. } => {
            Ok(format!("(({row}) * 16u + ({col}))"))
        }

        DType::MmaA {
            layout: TensorLayout::ColMajor,
            ..
        }
        | DType::MmaB {
            layout: TensorLayout::ColMajor,
            ..
        } => {
            Ok(format!("(({col}) * 16u + ({row}))"))
        }

        _ => Err(Error {
            msg: "expected MMA fragment dtype",
            kind: ErrorKind::InvalidDType,
            ctx: (),
        }),
    }
}

fn render_val(id: ValueId, kernel: &RawKernel, options: &CompilationOptions) -> Result<String, Error> {
    codegen::render_val(id, kernel, options, process_op)
}
