use crate::dispatch::backend::{
    Axis, DType, MetaId, ParamId, SharedId, ValueId, ValueState, kernel::RawKernel,
};

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Op {
    Nop,

    DefineVar {
        id: ValueId,
    },
    OverwriteVar {
        id: ValueId,
        val: ValueId,
    },
    AddAssign {
        id: ValueId,
        val: ValueId,
    },
    MulAssign {
        id: ValueId,
        val: ValueId,
    },
    DivAssign {
        id: ValueId,
        val: ValueId,
    },
    SubAssign {
        id: ValueId,
        val: ValueId,
    },
    ShlAssign {
        id: ValueId,
        val: ValueId,
    },
    ShrAssign {
        id: ValueId,
        val: ValueId,
    },
    CopyVar {
        id: ValueId,
    },

    ConstF64 {
        val: f64,
    },
    ConstF32 {
        val: f32,
    },
    ConstF16 {
        val: half::f16,
    },
    ConstBf16 {
        val: half::bf16,
    },
    ConstU32 {
        val: u32,
    },
    ConstI32 {
        val: i32,
    },

    ReadMeta {
        param: ParamId,
        field: MetaId,
    },

    LocalId {
        axis: Axis,
    },
    BlockId {
        axis: Axis,
    },
    GlobalId {
        axis: Axis,
    },

    Add {
        a: ValueId,
        b: ValueId,
    },
    Sub {
        a: ValueId,
        b: ValueId,
    },
    Mul {
        a: ValueId,
        b: ValueId,
    },
    Div {
        a: ValueId,
        b: ValueId,
    },
    Mod {
        a: ValueId,
        b: ValueId,
    },
    Pow {
        a: ValueId,
        b: ValueId,
    },

    Shl {
        a: ValueId,
        b: ValueId,
    },

    Shr {
        a: ValueId,
        b: ValueId,
    },

    /// (Fused) Operation `a * b + c`
    Fma {
        a: ValueId,
        b: ValueId,
        c: ValueId,
    },

    Exp {
        x: ValueId,
    },
    Abs {
        x: ValueId,
    },
    Neg {
        x: ValueId,
    },
    Log {
        x: ValueId,
    },
    Tanh {
        x: ValueId,
    },
    Sqrt {
        x: ValueId,
    },

    ParamLoad {
        param: ParamId,
        index: ValueId,
    },
    ParamStore {
        param: ParamId,
        index: ValueId,
        val: ValueId,
    },
    ParamAccum {
        param: ParamId,
        index: ValueId,
        val: ValueId,
    },
    ParamMul {
        param: ParamId,
        index: ValueId,
        val: ValueId,
    },
    ParamDiv {
        param: ParamId,
        index: ValueId,
        val: ValueId,
    },
    ParamSub {
        param: ParamId,
        index: ValueId,
        val: ValueId,
    },
    ParamShl {
        param: ParamId,
        index: ValueId,
        val: ValueId,
    },
    ParamShr {
        param: ParamId,
        index: ValueId,
        val: ValueId,
    },

    SharedLoad {
        mem: SharedId,
        index: ValueId,
    },
    SharedStore {
        mem: SharedId,
        index: ValueId,
        val: ValueId,
    },
    SharedAccum {
        mem: SharedId,
        index: ValueId,
        val: ValueId,
    },
    SharedMul {
        mem: SharedId,
        index: ValueId,
        val: ValueId,
    },
    SharedDiv {
        mem: SharedId,
        index: ValueId,
        val: ValueId,
    },
    SharedSub {
        mem: SharedId,
        index: ValueId,
        val: ValueId,
    },
    SharedShl {
        mem: SharedId,
        index: ValueId,
        val: ValueId,
    },
    SharedShr {
        mem: SharedId,
        index: ValueId,
        val: ValueId,
    },

    Eq {
        a: ValueId,
        b: ValueId,
    },
    Ne {
        a: ValueId,
        b: ValueId,
    },
    Lt {
        a: ValueId,
        b: ValueId,
    },
    Gt {
        a: ValueId,
        b: ValueId,
    },
    Le {
        a: ValueId,
        b: ValueId,
    },
    Ge {
        a: ValueId,
        b: ValueId,
    },

    Not {
        cond: ValueId,
    },

    Max {
        a: ValueId,
        b: ValueId,
    },
    Min {
        a: ValueId,
        b: ValueId,
    },

    Cast {
        id: ValueId,
        dtype: DType,
    },
    BitCast {
        id: ValueId,
        dtype: DType,
    },

    Select {
        cond: ValueId,
        a: ValueId,
        b: ValueId,
    },

    ForLoopBegin {
        index: ValueId,
        end: ValueId,
        step: ValueId,
    },

    ForeverLoopBegin,

    Continue,
    Break,

    IfBegin {
        cond: ValueId,
    },
    ElseBegin,

    StartScope,
    EndScope,

    Barrier,

    Return,

    /// Index into a generic pointer or array
    ///
    /// Exact semantics are undefined but you can assume:
    ///
    /// 1. indexing into a fragment is valid with unspecified order
    /// 2. indexing into a parameter or shared memory is always invalid
    Index {
        pointer: ValueId,
        index: ValueId,
    },

    /// Tensor/matrix core requires `TargetCompilationOptions::LIN_ACC`
    MmaSync {
        acc: ValueId,
        a: ValueId,
        b: ValueId,
        c: ValueId,
    },

    /// Tensor/matrix core requires `TargetCompilationOptions::LIN_ACC`
    LoadMatrixSync {
        frag: ValueId,
        index: ValueId,
        stride: ValueId,
    },

    /// Tensor/matrix core requires `TargetCompilationOptions::LIN_ACC`
    StoreMatrixSync {
        frag: ValueId,
        index: ValueId,
        stride: ValueId,
    },

    /// Tensor/matrix core requires `TargetCompilationOptions::LIN_ACC`
    FillFragment {
        frag: ValueId,
        val: ValueId,
    },
}

impl Op {
    #[must_use]
    #[inline]
    pub fn does_read(&self, value_id: ValueId) -> bool {
        match self {
            Self::Barrier
            | Self::BlockId { .. }
            | Self::Break
            | Self::ConstF64 { .. }
            | Self::ConstF32 { .. }
            | Self::ConstF16 { .. }
            | Self::ConstBf16 { .. }
            | Self::ConstI32 { .. }
            | Self::ConstU32 { .. }
            | Self::Continue
            | Self::DefineVar { .. }
            | Self::ElseBegin
            | Self::EndScope
            | Self::ForeverLoopBegin
            | Self::GlobalId { .. }
            | Self::LocalId { .. }
            | Self::ReadMeta { .. }
            | Self::Return
            | Self::Nop
            | Self::StartScope => false,
            Self::Abs { x }
            | Self::Exp { x }
            | Self::Log { x }
            | Self::Neg { x }
            | Self::Sqrt { x }
            | Self::Tanh { x }
            | Self::Not { cond: x }
            | Self::CopyVar { id: x }
            | Self::Cast { id: x, .. }
            | Self::BitCast { id: x, .. } => x == &value_id,
            Self::Add { a, b }
            | Self::Div { a, b }
            | Self::Eq { a, b }
            | Self::Ge { a, b }
            | Self::Gt { a, b }
            | Self::Le { a, b }
            | Self::Lt { a, b }
            | Self::Ne { a, b }
            | Self::Max { a, b }
            | Self::Min { a, b }
            | Self::Mod { a, b }
            | Self::Mul { a, b }
            | Self::Pow { a, b }
            | Self::Sub { a, b }
            | Self::Shl { a, b }
            | Self::Shr { a, b } => a == &value_id || b == &value_id,
            Self::Fma { a, b, c } => a == &value_id || b == &value_id || c == &value_id,
            Self::AddAssign { val, .. }
            | Self::DivAssign { val, .. }
            | Self::MulAssign { val, .. }
            | Self::ShlAssign { val, .. }
            | Self::ShrAssign { val, .. }
            | Self::SubAssign { val, .. }
            | Self::OverwriteVar { val, .. } => val == &value_id,
            Self::ForLoopBegin { index, end, step } => index == &value_id || end == &value_id || step == &value_id,
            Self::IfBegin { cond } => cond == &value_id,
            Self::ParamAccum { index, val, .. }
            | Self::ParamDiv { index, val, .. }
            | Self::ParamMul { index, val, .. }
            | Self::ParamShl { index, val, .. }
            | Self::ParamShr { index, val, .. }
            | Self::ParamStore { index, val, .. }
            | Self::ParamSub { index, val, .. }
            | Self::SharedAccum { index, val, .. }
            | Self::SharedDiv { index, val, .. }
            | Self::SharedMul { index, val, .. }
            | Self::SharedShl { index, val, .. }
            | Self::SharedShr { index, val, .. }
            | Self::SharedStore { index, val, .. }
            | Self::SharedSub { index, val, .. } => index == &value_id || val == &value_id,
            Self::ParamLoad { index, .. } | Self::SharedLoad { index, .. } => index == &value_id,
            Self::Select { cond, a, b } => cond == &value_id || a == &value_id || b == &value_id,
            Self::MmaSync { acc, a, b, c } => a == &value_id || b == &value_id || c == &value_id || acc == &value_id,
            Self::Index { pointer, index } => pointer == &value_id || index == &value_id,
            Self::LoadMatrixSync {
                frag,
                index,
                stride,
            }
            | Self::StoreMatrixSync {
                frag,
                index,
                stride,
            } => index == &value_id || frag == &value_id || stride == &value_id,
            Self::FillFragment { frag, val } => frag == &value_id || val == &value_id,
        }
    }

    #[inline]
    pub const fn replace_usage(&mut self, old_id: ValueId, new_id: ValueId) {
        #[inline]
        const fn replace_if_eq(value_id: &mut ValueId, old_id: ValueId, new_id: ValueId) {
            if *value_id == old_id {
                *value_id = new_id;
            }
        }

        match self {
            Self::CopyVar { id } => replace_if_eq(id, old_id, new_id),
            Self::Abs { x }
            | Self::Exp { x }
            | Self::Log { x }
            | Self::Neg { x }
            | Self::Sqrt { x }
            | Self::Tanh { x }
            | Self::Not { cond: x }
            | Self::Cast { id: x, .. }
            | Self::BitCast { id: x, .. } => replace_if_eq(x, old_id, new_id),
            Self::Add { a, b }
            | Self::Div { a, b }
            | Self::Eq { a, b }
            | Self::Ge { a, b }
            | Self::Gt { a, b }
            | Self::Le { a, b }
            | Self::Lt { a, b }
            | Self::Ne { a, b }
            | Self::Max { a, b }
            | Self::Min { a, b }
            | Self::Mod { a, b }
            | Self::Mul { a, b }
            | Self::Pow { a, b }
            | Self::Sub { a, b }
            | Self::Shl { a, b }
            | Self::Shr { a, b } => {
                replace_if_eq(a, old_id, new_id);
                replace_if_eq(b, old_id, new_id);
            }
            Self::Fma { a, b, c } => {
                replace_if_eq(a, old_id, new_id);
                replace_if_eq(b, old_id, new_id);
                replace_if_eq(c, old_id, new_id);
            }
            Self::AddAssign { val, id }
            | Self::DivAssign { val, id }
            | Self::MulAssign { val, id }
            | Self::ShlAssign { val, id }
            | Self::ShrAssign { val, id }
            | Self::SubAssign { val, id }
            | Self::OverwriteVar { val, id } => {
                replace_if_eq(id, old_id, new_id);
                replace_if_eq(val, old_id, new_id);
            }
            Self::ForLoopBegin { index, end, step } => {
                replace_if_eq(index, old_id, new_id);
                replace_if_eq(end, old_id, new_id);
                replace_if_eq(step, old_id, new_id);
            }
            Self::IfBegin { cond } => replace_if_eq(cond, old_id, new_id),
            Self::ParamAccum { index, val, .. }
            | Self::ParamDiv { index, val, .. }
            | Self::ParamMul { index, val, .. }
            | Self::ParamShl { index, val, .. }
            | Self::ParamShr { index, val, .. }
            | Self::ParamStore { index, val, .. }
            | Self::ParamSub { index, val, .. }
            | Self::SharedAccum { index, val, .. }
            | Self::SharedDiv { index, val, .. }
            | Self::SharedMul { index, val, .. }
            | Self::SharedShl { index, val, .. }
            | Self::SharedShr { index, val, .. }
            | Self::SharedStore { index, val, .. }
            | Self::SharedSub { index, val, .. } => {
                replace_if_eq(index, old_id, new_id);
                replace_if_eq(val, old_id, new_id);
            }
            Self::ParamLoad { index, .. } | Self::SharedLoad { index, .. } => {
                replace_if_eq(index, old_id, new_id);
            }
            Self::Select { cond, a, b } => {
                replace_if_eq(cond, old_id, new_id);
                replace_if_eq(a, old_id, new_id);
                replace_if_eq(b, old_id, new_id);
            }
            Self::MmaSync { a, b, c, acc } => {
                replace_if_eq(a, old_id, new_id);
                replace_if_eq(b, old_id, new_id);
                replace_if_eq(c, old_id, new_id);
                replace_if_eq(acc, old_id, new_id);
            }
            Self::Index { pointer, index } => {
                replace_if_eq(pointer, old_id, new_id);
                replace_if_eq(index, old_id, new_id);
            }
            Self::LoadMatrixSync {
                frag,
                index,
                stride,
            } => {
                replace_if_eq(frag, old_id, new_id);
                replace_if_eq(index, old_id, new_id);
                replace_if_eq(stride, old_id, new_id);
            }
            Self::StoreMatrixSync {
                index,
                frag,
                stride,
            } => {
                replace_if_eq(index, old_id, new_id);
                replace_if_eq(frag, old_id, new_id);
                replace_if_eq(stride, old_id, new_id);
            }
            Self::FillFragment { frag, val } => {
                replace_if_eq(frag, old_id, new_id);
                replace_if_eq(val, old_id, new_id);
            }
            Self::Barrier
            | Self::BlockId { .. }
            | Self::Break
            | Self::ConstF64 { .. }
            | Self::ConstF32 { .. }
            | Self::ConstF16 { .. }
            | Self::ConstBf16 { .. }
            | Self::ConstI32 { .. }
            | Self::ConstU32 { .. }
            | Self::Continue
            | Self::DefineVar { .. }
            | Self::ElseBegin
            | Self::EndScope
            | Self::ForeverLoopBegin
            | Self::GlobalId { .. }
            | Self::LocalId { .. }
            | Self::ReadMeta { .. }
            | Self::Return
            | Self::Nop
            | Self::StartScope => {}
        }
    }

    #[inline]
    #[must_use]
    pub fn read_only(&self, kernel: &RawKernel) -> bool {
        #[inline]
        fn not_mut(value_id: ValueId, kernel: &RawKernel) -> bool {
            kernel.vals[value_id].state != ValueState::Mut
        }

        match self {
            Self::CopyVar { id } => not_mut(*id, kernel),
            Self::Abs { x }
            | Self::Exp { x }
            | Self::Log { x }
            | Self::Neg { x }
            | Self::Sqrt { x }
            | Self::Tanh { x }
            | Self::Not { cond: x }
            | Self::Cast { id: x, .. }
            | Self::BitCast { id: x, .. } => not_mut(*x, kernel),
            Self::Add { a, b }
            | Self::Div { a, b }
            | Self::Eq { a, b }
            | Self::Ge { a, b }
            | Self::Gt { a, b }
            | Self::Le { a, b }
            | Self::Lt { a, b }
            | Self::Ne { a, b }
            | Self::Max { a, b }
            | Self::Min { a, b }
            | Self::Mod { a, b }
            | Self::Mul { a, b }
            | Self::Pow { a, b }
            | Self::Sub { a, b }
            | Self::Shl { a, b }
            | Self::Shr { a, b } => not_mut(*a, kernel) && not_mut(*b, kernel),
            Self::Fma { a, b, c } => not_mut(*a, kernel) && not_mut(*b, kernel) && not_mut(*c, kernel),
            Self::AddAssign { val, id }
            | Self::DivAssign { val, id }
            | Self::MulAssign { val, id }
            | Self::ShlAssign { val, id }
            | Self::ShrAssign { val, id }
            | Self::SubAssign { val, id }
            | Self::OverwriteVar { val, id } => not_mut(*val, kernel) && not_mut(*id, kernel),
            Self::ForLoopBegin { index, end, step } => not_mut(*index, kernel) && not_mut(*end, kernel) && not_mut(*step, kernel),
            Self::IfBegin { cond } => not_mut(*cond, kernel),
            Self::ParamAccum { index, val, .. }
            | Self::ParamDiv { index, val, .. }
            | Self::ParamMul { index, val, .. }
            | Self::ParamShl { index, val, .. }
            | Self::ParamShr { index, val, .. }
            | Self::ParamStore { index, val, .. }
            | Self::ParamSub { index, val, .. }
            | Self::SharedAccum { index, val, .. }
            | Self::SharedDiv { index, val, .. }
            | Self::SharedMul { index, val, .. }
            | Self::SharedShl { index, val, .. }
            | Self::SharedShr { index, val, .. }
            | Self::SharedStore { index, val, .. }
            | Self::SharedSub { index, val, .. } => not_mut(*index, kernel) && not_mut(*val, kernel),
            Self::ParamLoad { index, .. } | Self::SharedLoad { index, .. } => not_mut(*index, kernel),
            Self::Select { cond, a, b } => not_mut(*cond, kernel) && not_mut(*a, kernel) && not_mut(*b, kernel),
            Self::MmaSync { a, b, c, acc } => not_mut(*a, kernel)
                && not_mut(*b, kernel)
                && not_mut(*c, kernel)
                && not_mut(*acc, kernel),
            Self::Index { pointer, index } => not_mut(*pointer, kernel) && not_mut(*index, kernel),
            Self::LoadMatrixSync {
                frag,
                index,
                stride,
            }
            | Self::StoreMatrixSync {
                frag,
                index,
                stride,
            } => not_mut(*frag, kernel) && not_mut(*index, kernel) && not_mut(*stride, kernel),
            Self::FillFragment { frag, val } => not_mut(*frag, kernel) && not_mut(*val, kernel),
            Self::Barrier
            | Self::BlockId { .. }
            | Self::Break
            | Self::ConstF64 { .. }
            | Self::ConstF32 { .. }
            | Self::ConstF16 { .. }
            | Self::ConstBf16 { .. }
            | Self::ConstI32 { .. }
            | Self::ConstU32 { .. }
            | Self::Continue
            | Self::DefineVar { .. }
            | Self::ElseBegin
            | Self::EndScope
            | Self::ForeverLoopBegin
            | Self::GlobalId { .. }
            | Self::LocalId { .. }
            | Self::ReadMeta { .. }
            | Self::Return
            | Self::Nop
            | Self::StartScope => false,
        }
    }

    #[inline]
    #[must_use]
    pub fn does_write(&self, value_id: ValueId) -> bool {
        match self {
            Self::DefineVar { id }
            | Self::AddAssign { id, .. }
            | Self::DivAssign { id, .. }
            | Self::MulAssign { id, .. }
            | Self::ShlAssign { id, .. }
            | Self::ShrAssign { id, .. }
            | Self::SubAssign { id, .. }
            | Self::OverwriteVar { id, .. }
            | Self::ForLoopBegin { index: id, .. }
            | Self::FillFragment { frag: id, .. }
            | Self::LoadMatrixSync { frag: id, .. }
            | Self::SharedStore { mem: id, .. }
            | Self::MmaSync { acc: id, .. } => id == &value_id,
            _ => false,
        }
    }

    #[inline]
    #[must_use]
    pub const fn writes_to(&self) -> Option<ValueId> {
        match self {
            Self::DefineVar { id }
            | Self::AddAssign { id, .. }
            | Self::DivAssign { id, .. }
            | Self::MulAssign { id, .. }
            | Self::ShlAssign { id, .. }
            | Self::ShrAssign { id, .. }
            | Self::SubAssign { id, .. }
            | Self::OverwriteVar { id, .. }
            | Self::ForLoopBegin { index: id, .. }
            | Self::FillFragment { frag: id, .. }
            | Self::LoadMatrixSync { frag: id, .. }
            | Self::SharedStore { mem: id, .. }
            | Self::MmaSync { acc: id, .. } => Some(*id),
            _ => None,
        }
    }

    #[inline]
    #[must_use]
    pub fn does_mutate(&self, value_id: ValueId) -> bool {
        match self {
            Self::AddAssign { id, .. }
            | Self::DivAssign { id, .. }
            | Self::MulAssign { id, .. }
            | Self::ShlAssign { id, .. }
            | Self::ShrAssign { id, .. }
            | Self::SubAssign { id, .. }
            | Self::OverwriteVar { id, .. }
            | Self::ForLoopBegin { index: id, .. }
            | Self::FillFragment { frag: id, .. }
            | Self::LoadMatrixSync { frag: id, .. }
            | Self::SharedStore { mem: id, .. }
            | Self::MmaSync { acc: id, .. } => id == &value_id,
            _ => false,
        }
    }

    #[inline]
    #[must_use]
    pub fn is_zero(&self) -> bool {
        match self {
            Self::ConstI32 { val } => *val == 0,
            Self::ConstU32 { val } => *val == 0,
            Self::ConstF32 { val } => val.abs() < 1e-9,
            Self::ConstBf16 { val } => val.to_f32().abs() < 1e-9,
            Self::ConstF16 { val } => val.to_f32().abs() < 1e-9,
            Self::ConstF64 { val } => val.abs() < 1e-9,
            _ => false,
        }
    }

    #[inline]
    #[must_use]
    pub fn is_one(&self) -> bool {
        match self {
            Self::ConstI32 { val } => *val == 1,
            Self::ConstU32 { val } => *val == 1,
            Self::ConstF32 { val } => (val - 1.0).abs() < 1e-9,
            Self::ConstBf16 { val } => (val.to_f32() - 1.0).abs() < 1e-9,
            Self::ConstF16 { val } => (val.to_f32() - 1.0).abs() < 1e-9,
            Self::ConstF64 { val } => (val - 1.0).abs() < 1e-9,
            _ => false,
        }
    }
}
