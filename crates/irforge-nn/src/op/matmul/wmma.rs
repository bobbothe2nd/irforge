use fused_gpu::{
    dispatch::{
        CompilationOptions, backend::{
            DType, Graph, NodeId, Op, Param, ParamId, SimpleDType, TensorLayout, ValueId, ValueState, kernel::{LinkedKernel, NodeInput},
        },
    }, errors::{Error, ErrorKind},
};
use std::vec::Vec;

pub fn forward_matmul(
    _eval_node: impl Fn(
        NodeId,
        &NodeInput,
        ValueId,
        &mut Vec<NodeId>,
        &Graph,
        &[Option<ParamId>],
        &[Option<ParamId>],
        &mut LinkedKernel,
        ValueId,
        ValueId,
        ValueId,
        ValueId,
        u32,
        ValueId,
        &[Param],
        &mut bool,
        &CompilationOptions,
    ) -> Result<Vec<NodeId>, Error>,
    dtype: SimpleDType,
    _root: NodeId,
    _resolved: &mut Vec<NodeId>,
    _m: ValueId,
    n: ValueId,
    k: ValueId,
    swap_a: bool,
    swap_b: bool,
    node_id: NodeId,
    a_node: &NodeInput,
    b_node: &NodeInput,
    _graph: &Graph,
    _out: ValueId,
    node_params: &[Option<ParamId>],
    _saved_params: &[Option<ParamId>],
    kernel: &mut LinkedKernel,
    base: ValueId,
    row: ValueId,
    col: ValueId,
    _local_row: ValueId,
    _local_col: ValueId,
    _shared_size: u32,
    tile_size: ValueId,
    _params: &[Param],
    _stable_iteration_space: &mut bool,
    options: &CompilationOptions,
) -> Result<Vec<NodeId>, Error> {
    let subgroup_size = options.target.subgroup_size.ok_or(Error {
        msg: "attempted WMMA matmul with unknown subgroup size",
        kind: ErrorKind::InternalError,
        ctx: (),
    })?.get() as u32;
    let block = [subgroup_size / 8, 8, 1];

    let tile_width = options.opt.tile_size;

    kernel.raw.block.tile = [tile_width, tile_width, 1];
    kernel.raw.block.dim3 = block;

    let a = match a_node {
        NodeInput::Param { param, .. } => *param,
        NodeInput::Node(node_id) => node_params[*node_id].ok_or(Error {
            msg: "expected a_node mapping to param",
            kind: ErrorKind::UnresolvedInput,
            ctx: (),
        })?,
    };
    let b = match b_node {
        NodeInput::Param { param, .. } => *param,
        NodeInput::Node(node_id) => node_params[*node_id].ok_or(Error {
            msg: "expected a_node mapping to param",
            kind: ErrorKind::UnresolvedInput,
            ctx: (),
        })?,
    };
    let c = node_params[node_id].ok_or(Error {
        msg: "expected a_node mapping to param",
        kind: ErrorKind::UnresolvedOutput,
        ctx: (),
    })?;

    let deepest = Vec::new();

    let frag_a = kernel.raw.def_var(
        DType::MmaA {
            dtype,
            m: tile_width,
            n: tile_width,
            k: tile_width,
            layout: if swap_a {
                TensorLayout::ColMajor
            } else { 
                TensorLayout::RowMajor
            },
        },
        ValueState::Mut,
        None,
    );
    let frag_b = kernel.raw.def_var(
        DType::MmaB {
            dtype,
            m: tile_width,
            n: tile_width,
            k: tile_width,
            layout: if swap_b {
                TensorLayout::ColMajor
            } else { 
                TensorLayout::RowMajor
            },
        },
        ValueState::Mut,
        None,
    );
    let frag_c = kernel.raw.def_var(
        DType::MmaAccum { dtype, m: tile_width, n: tile_width, k: tile_width },
        ValueState::Mut,
        None,
    );

    let zero = kernel.raw.def_var(
        DType::Simple(dtype),
        ValueState::Inline,
        Some(Op::ConstF32 { val: 0.0 }),
    );
    kernel.raw.fill_frag(frag_c, zero);

    let tk = kernel.raw.def_var(
        DType::Simple(SimpleDType::U32),
        ValueState::Mut,
        Some(Op::ConstU32 { val: 0 }),
    );

    kernel.push_for_loop(tk, k, tile_size, |kernel| {
        let a_col = if swap_a {
            let a_row = kernel.raw.def_var(DType::Simple(SimpleDType::U32), ValueState::Inline, Some(Op::Mul { a: k, b: tk }));
            kernel.raw.def_var(DType::Simple(SimpleDType::U32), ValueState::Inline, Some(Op::Add { a: a_row, b: row }))
        } else {
            let a_row = kernel.raw.def_var(DType::Simple(SimpleDType::U32), ValueState::Inline, Some(Op::Mul { a: k, b: row }));
            kernel.raw.def_var(DType::Simple(SimpleDType::U32), ValueState::Inline, Some(Op::Add { a: a_row, b: tk }))
        };
        let a_idx = kernel.raw.def_var(DType::Simple(SimpleDType::U32), ValueState::Inline, Some(Op::Add { a: a_col, b: base }));

        kernel.raw.load_matrix_sync(a, frag_a, a_idx, k);

        let b_col = if swap_b {
            let b_row = kernel.raw.def_var(DType::Simple(SimpleDType::U32), ValueState::Inline, Some(Op::Mul { a: n, b: col }));
            kernel.raw.def_var(DType::Simple(SimpleDType::U32), ValueState::Inline, Some(Op::Add { a: b_row, b: tk }))
        } else {
            let b_row = kernel.raw.def_var(DType::Simple(SimpleDType::U32), ValueState::Inline, Some(Op::Mul { a: n, b: tk }));
            kernel.raw.def_var(DType::Simple(SimpleDType::U32), ValueState::Inline, Some(Op::Add { a: b_row, b: col }))
        };
        let b_idx = kernel.raw.def_var(DType::Simple(SimpleDType::U32), ValueState::Inline, Some(Op::Add { a: b_col, b: base }));

        kernel.raw.load_matrix_sync(b, frag_b, b_idx, n);

        kernel.raw.mma_sync(frag_c, frag_a, frag_b, frag_c);

        Ok(())
    })?;

    let c_row = kernel.raw.def_var(DType::Simple(SimpleDType::U32), ValueState::Inline, Some(Op::Mul { a: row, b: n }));
    let c_col = kernel.raw.def_var(DType::Simple(SimpleDType::U32), ValueState::Inline, Some(Op::Add { a: c_row, b: col }));
    let c_idx = kernel.raw.def_var(DType::Simple(SimpleDType::U32), ValueState::Inline, Some(Op::Add { a: c_col, b: base }));

    kernel.raw.store_matrix_sync(c, frag_c, c_idx, n, TensorLayout::RowMajor);

    Ok(deepest)
}
