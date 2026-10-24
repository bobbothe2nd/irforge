use crate::{
    dispatch::{
        CompilationOptions, OptFlags,
        backend::{Op, ValueId, ValueState, kernel::RawKernel},
    },
    errors::Error,
};
use std::vec::Vec;

#[allow(clippy::unnecessary_wraps)]
pub fn optimize(kernel: &mut RawKernel, options: &CompilationOptions) -> Result<(), Error> {
    for _ in 0..options.opt.passes {
        dead_code_elimination(kernel, options);

        if options.opt.flags.contains(OptFlags::UNUSED_MUT) {
            for val_id in 0..kernel.vals.len() {
                if kernel.vals[val_id].state == ValueState::Mut
                    && kernel.vals[val_id].init.is_some()
                    && !(0..kernel.ops.len())
                        .any(|statement| kernel.ops[statement].does_mutate(val_id))
                {
                    kernel.vals[val_id].state = ValueState::Immut;
                }
            }
        }

        if options.opt.flags.contains(OptFlags::OP_ASSIGN) {
            op_assign(kernel);
        }

        if options.opt.flags.contains(OptFlags::MUL_ADD) {
            for val_id in 0..kernel.vals.len() {
                if kernel.vals[val_id].state == ValueState::Masked
                    || !kernel.vals[val_id].dtype.is_float()
                {
                    continue;
                }

                if let Some(Op::Add { a, b }) = kernel.vals[val_id].init {
                    if let Some(Op::Mul { a: a_mul, b: b_mul }) = kernel.vals[a].init {
                        kernel.vals[val_id].init.replace(Op::Fma {
                            a: a_mul,
                            b: b_mul,
                            c: b,
                        });
                    } else if let Some(Op::Mul { a: a_mul, b: b_mul }) = kernel.vals[b].init {
                        kernel.vals[val_id].init.replace(Op::Fma {
                            a: a_mul,
                            b: b_mul,
                            c: a,
                        });
                    }
                }
            }

            let additions = iter_vals(kernel, |op, val_id| {
                if kernel.vals[val_id].state == ValueState::Masked
                    || !kernel.vals[val_id].dtype.is_float()
                {
                    return None;
                }

                if let Op::Add { a, b } = &op {
                    Some((*a, *b, val_id))
                } else {
                    None
                }
            });

            for (a, b, val_id) in additions {
                if let Some(Op::Mul { a: a_mul, b: b_mul }) = kernel.vals[a].init {
                    kernel.vals[val_id].init.replace(Op::Fma {
                        a: a_mul,
                        b: b_mul,
                        c: b,
                    });
                }

                if let Some(Op::Mul { a: a_mul, b: b_mul }) = kernel.vals[b].init {
                    kernel.vals[val_id].init.replace(Op::Fma {
                        a: a_mul,
                        b: b_mul,
                        c: a,
                    });
                }
            }
        }

        if options.opt.flags.contains(OptFlags::DIV_CONST) {
            let divisions = iter_vals(kernel, |op, val_id| {
                if kernel.vals[val_id].state == ValueState::Masked
                    || !kernel.vals[val_id].dtype.is_float()
                {
                    return None;
                }

                if let Op::Div { a, b } = &op {
                    Some((*a, *b, val_id))
                } else {
                    None
                }
            });

            for (a, b, val_id) in divisions {
                if kernel.vals[b].state == ValueState::Mut {
                    continue;
                }

                if let Some(Op::ConstF32 { val }) = kernel.vals[b].init {
                    kernel.vals[b]
                        .init
                        .replace(Op::ConstF32 { val: 1.0 / val });

                    kernel.vals[val_id].init.replace(Op::Mul { a, b });
                }
            }
        }

        if options.opt.flags.contains(OptFlags::CONST_FOLD) {
            for val_id in 0..kernel.vals.len() {
                if kernel.vals[val_id].state == ValueState::Masked {
                    continue;
                }

                const_fold(kernel, val_id);
            }
        }

        if options.opt.flags.contains(OptFlags::IDENTITY) {
            for val_id in 0..kernel.vals.len() {
                if kernel.vals[val_id].state == ValueState::Masked {
                    continue;
                }

                identity(kernel, val_id);
            }
        }

        if options.opt.flags.contains(OptFlags::COPY_IMMUT) {
            for val_id in 0..kernel.vals.len() {
                if !matches!(
                    kernel.vals[val_id].state,
                    ValueState::Inline | ValueState::Immut
                ) {
                    continue;
                }

                let copy_op = Op::CopyVar { id: val_id };

                let copies = iter_vals(kernel, |op, val_id| {
                    if op == &copy_op && kernel.vals[val_id].state != ValueState::Mut {
                        Some(val_id)
                    } else {
                        None
                    }
                });

                for copy_id in copies {
                    for val in &mut kernel.vals {
                        if let Some(op) = &mut val.init {
                            op.replace_usage(copy_id, val_id);
                        }
                    }

                    for op in &mut kernel.ops {
                        op.replace_usage(copy_id, val_id);
                    }
                }
            }
        }

        // if options.opt.flags.contains(OptFlags::LOOP_STATELESS) {}

        // if options.opt.flags.contains(OptFlags::EMPTY_SCOPE) {}

        // if options.opt.flags.contains(OptFlags::LATE_INIT) {}

        kernel.ops = erase_nops(&kernel.ops);
    }

    dead_code_elimination(kernel, options);

    Ok(())
}

#[inline]
fn dead_code_elimination(kernel: &mut RawKernel, options: &CompilationOptions) {
    if options.opt.flags.contains(OptFlags::DEAD_CODE) {
        for val_id in 0..kernel.vals.len() {
            if kernel.vals[val_id].state == ValueState::Masked {
                continue;
            }

            if !any_ops(kernel, |op| {
                op.does_read(val_id) && !op.does_mutate(val_id)
            }) {
                kernel.vals[val_id].state = ValueState::Masked;
                erase_writes(kernel, val_id);
            }
        }
    }
}

#[inline]
fn iter_vals<R>(kernel: &RawKernel, mut f: impl FnMut(&Op, ValueId) -> Option<R>) -> Vec<R> {
    let mut val_ids = Vec::new();

    for val_id in 0..kernel.vals.len() {
        if kernel.vals[val_id].state == ValueState::Masked {
            continue;
        }

        if let Some(op) = &kernel.vals[val_id].init
            && let Some(val) = f(op, val_id)
        {
            val_ids.push(val);
        }
    }

    val_ids
}

#[inline]
fn any_ops(kernel: &RawKernel, mut f: impl FnMut(&Op) -> bool) -> bool {
    for op in &kernel.ops {
        if f(op) {
            return true;
        }
    }

    for val in &kernel.vals {
        if let Some(init) = &val.init
            && f(init)
        {
            return true;
        }
    }

    false
}

#[inline]
fn erase_writes(kernel: &mut RawKernel, val_id: ValueId) {
    for op in &mut kernel.ops {
        if op.does_write(val_id) {
            *op = Op::Nop;
        }
    }
}

#[inline]
fn erase_nops(ops: &[Op]) -> Vec<Op> {
    let mut new_ops = Vec::new();

    for op in ops {
        if *op != Op::Nop {
            new_ops.push(*op);
        }
    }

    new_ops
}

macro_rules! const_fold_op {
    ($kernel:ident, $val_id:ident, $a:ident, $b:expr, $opu:path, $opi:path, $($opf:path)?) => {{
        let a = $kernel.vals[*$a];
        let b = $kernel.vals[*$b];

        if a.state == ValueState::Mut
        || b.state == ValueState::Mut {
            return;
        }

        if let Some(a_op) = a.init
        && let Some(b_op) = b.init {
            match (a_op, b_op) {
                $((Op::ConstF32 { val: a }, Op::ConstF32 { val: b }) => {
                    $kernel.vals[$val_id].init.replace(Op::ConstF32 { val: $opf(a, b) });
                })?
                (Op::ConstU32 { val: a }, Op::ConstU32 { val: b }) => {
                    $kernel.vals[$val_id].init.replace(Op::ConstU32 { val: $opu(a, b) });
                }
                (Op::ConstI32 { val: a }, Op::ConstI32 { val: b }) => {
                    $kernel.vals[$val_id].init.replace(Op::ConstI32 { val: $opi(a, b) });
                }
                _ => {}
            }
        }
    }};
}

#[inline]
fn const_fold(kernel: &mut RawKernel, val_id: ValueId) {
    use core::ops::{Add, Div, Mul, Shl, Shr, Sub};

    if let Some(init) = &kernel.vals[val_id].init {
        match init {
            Op::Add { a, b } => {
                const_fold_op!(kernel, val_id, a, b, u32::add, i32::add, f32::add);
            }
            Op::Mul { a, b } => {
                const_fold_op!(kernel, val_id, a, b, u32::mul, i32::mul, f32::mul);
            }
            Op::Sub { a, b } => {
                const_fold_op!(kernel, val_id, a, b, u32::sub, i32::sub, f32::sub);
            }
            Op::Div { a, b } => {
                const_fold_op!(kernel, val_id, a, b, u32::div, i32::div, f32::div);
            }
            Op::Shr { a, b } => const_fold_op!(kernel, val_id, a, b, u32::shr, i32::shr,),
            Op::Shl { a, b } => const_fold_op!(kernel, val_id, a, b, u32::shl, i32::shl,),
            Op::Fma { a, b, c } => {
                let a = kernel.vals[*a];
                let b = kernel.vals[*b];
                let c = kernel.vals[*c];

                if a.state == ValueState::Mut
                    || b.state == ValueState::Mut
                    || c.state == ValueState::Mut
                {
                    return;
                }

                if let Some(Op::ConstF32 { val: a }) = a.init
                    && let Some(Op::ConstF32 { val: b }) = b.init
                    && let Some(Op::ConstF32 { val: c }) = c.init
                {
                    kernel.vals[val_id].init.replace(Op::ConstF32 {
                        val: a.mul_add(b, c),
                    });
                }
            }
            _ => {}
        }
    }
}

#[inline]
fn identity(kernel: &mut RawKernel, val_id: ValueId) {
    if let Some(init) = kernel.vals[val_id].init {
        match init {
            Op::Add { a, b } => {
                if kernel.vals[a].init.is_some_and(|op| op.is_zero())
                    && kernel.vals[a].state != ValueState::Mut
                {
                    kernel.vals[val_id].init.replace(Op::CopyVar { id: b });
                } else if kernel.vals[b].init.is_some_and(|op| op.is_zero())
                    && kernel.vals[b].state != ValueState::Mut
                {
                    kernel.vals[val_id].init.replace(Op::CopyVar { id: a });
                }
            }
            Op::Sub { a, b } => {
                if kernel.vals[b].init.is_some_and(|op| op.is_zero())
                    && kernel.vals[b].state != ValueState::Mut
                {
                    kernel.vals[val_id].init.replace(Op::CopyVar { id: a });
                }
            }
            Op::Mul { a, b } => {
                if kernel.vals[a].init.is_some_and(|op| op.is_one())
                    && kernel.vals[a].state != ValueState::Mut
                {
                    kernel.vals[val_id].init.replace(Op::CopyVar { id: b });
                }
                if kernel.vals[b].init.is_some_and(|op| op.is_one())
                    && kernel.vals[b].state != ValueState::Mut
                {
                    kernel.vals[val_id].init.replace(Op::CopyVar { id: a });
                }
            }
            Op::Div { a, b }
                if kernel.vals[b].init.is_some_and(|op| op.is_one())
                    && kernel.vals[b].state != ValueState::Mut =>
            {
                kernel.vals[val_id].init.replace(Op::CopyVar { id: a });
            }
            _ => {}
        }
    }
}

macro_rules! impl_op_assign {
    (@var $kernel:ident, $op_id:ident, $id:ident, $val:ident, $($inline:ident => $assign:ident),*) => {
        match $val.init {
            $(
                Some(Op::$inline { a, b }) => if a == $id {
                    $kernel.ops[$op_id] = Op::$assign { $id, val: b }
                } else if b == $id {
                    $kernel.ops[$op_id] = Op::$assign { $id, val: a }
                }
            )*
            _ => {}
        }
    };

    (@param $load:ident, $kernel:ident, $op_id:ident, $param:ident, $index:ident, $val:ident, $($inline:ident => $assign:ident),*) => {
        match $val.init {
            $(
                Some(Op::$inline { a, b }) => if $kernel.vals[a].init == Some(Op::$load { $param, $index }) {
                    $kernel.ops[$op_id] = Op::$assign { $param, $index, val: b }
                } else if $kernel.vals[b].init == Some(Op::$load { $param, $index }) {
                    $kernel.ops[$op_id] = Op::$assign { $param, $index, val: a }
                }
            )*
            _ => {}
        }
    };
}

#[inline]
fn op_assign(kernel: &mut RawKernel) {
    for op_id in 0..kernel.ops.len() {
        if let Op::OverwriteVar { id, val } = kernel.ops[op_id] {
            let val = &mut kernel.vals[val];

            if val.state == ValueState::Inline {
                impl_op_assign!(
                    @var
                    kernel,
                    op_id,
                    id,
                    val,
                    Add => AddAssign,
                    Sub => SubAssign,
                    Mul => MulAssign,
                    Div => DivAssign,
                    Shr => ShrAssign,
                    Shl => ShlAssign
                );
            }
        }

        if let Op::ParamStore {
            param,
            index,
            val,
        } = kernel.ops[op_id]
        {
            let val = &mut kernel.vals[val];

            if val.state == ValueState::Inline {
                impl_op_assign!(
                    @param
                    ParamLoad,
                    kernel,
                    op_id,
                    param,
                    index,
                    val,
                    Add => ParamAccum,
                    Sub => ParamSub,
                    Mul => ParamMul,
                    Div => ParamDiv,
                    Shr => ParamShr,
                    Shl => ParamShl
                );
            }
        }

        if let Op::SharedStore { mem, index, val } = kernel.ops[op_id] {
            let val = &mut kernel.vals[val];

            if val.state == ValueState::Inline {
                impl_op_assign!(
                    @param
                    SharedLoad,
                    kernel,
                    op_id,
                    mem,
                    index,
                    val,
                    Add => SharedAccum,
                    Sub => SharedSub,
                    Mul => SharedMul,
                    Div => SharedDiv,
                    Shr => SharedShr,
                    Shl => SharedShl
                );
            }
        }
    }
}
