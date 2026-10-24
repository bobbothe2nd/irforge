use fused_gpu::{
    dispatch::{
        CompilationOptions,
        backend::{
            DType, Graph, NodeId, Op, Param, ParamId, SimpleDType, ValueId, ValueState,
            kernel::{LinkedKernel, NodeInput},
        },
    }, errors::{Error, ErrorKind},
};
use std::vec::Vec;

pub fn forward_matmul(
    _e: impl Fn(
        NodeId,
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
    _input: NodeId,
    _resolved: &mut Vec<NodeId>,
    _m: ValueId,
    n: ValueId,
    k: ValueId,
    _swap_a: bool,
    _swap_b: bool,
    _a_node: &NodeInput,
    _b_node: &NodeInput,
    _graph: &Graph,
    _out: ValueId,
    _node_params: &[Option<ParamId>],
    _saved_params: &[Option<ParamId>],
    kernel: &mut LinkedKernel,
    _base: ValueId,
    _row: ValueId,
    _col: ValueId,
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
    kernel.raw.block.dim3 = block;

    let tile_width = options.opt.tile_size;

    let deepest = Vec::new();

    let tile_m = kernel.raw.def_var(
        DType::Simple(SimpleDType::U32),
        ValueState::Inline,
        Some(Op::ConstU32 { val: 16 * block[0] }),
    );
    let tile_n = kernel.raw.def_var(
        DType::Simple(SimpleDType::U32),
        ValueState::Inline,
        Some(Op::ConstU32 { val: 16 * block[1] }),
    );

    let frag_a = kernel.raw.def_var(
        DType::MmaA { dtype, m: tile_width, n: tile_width, k: tile_width },
        ValueState::Mut,
        None,
    );
    let frag_b = kernel.raw.def_var(
        DType::MmaB { dtype, m: tile_width, n: tile_width, k: tile_width },
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
        DType::Simple(SimpleDType::F32),
        ValueState::Mut,
        Some(Op::ConstU32 { val: 0 }),
    );

    // let a_deepest = Vec::new();
    // let b_deepest = Vec::new();

    kernel.push_for_loop(tk, k, tile_size, |kernel| {
        let a_stride = kernel.raw.def_var(DType::Simple(SimpleDType::U32), ValueState::Inline, Some(Op::Mul { a: k, b: tile_m }));
        let b_stride = kernel.raw.def_var(DType::Simple(SimpleDType::U32), ValueState::Inline, Some(Op::Mul { a: tk, b: n }));

        let a_off = kernel.raw.def_var(DType::Simple(SimpleDType::U32), ValueState::Inline, Some(Op::Mul { a: a_stride, b: tk }));
        let b_off = kernel.raw.def_var(DType::Simple(SimpleDType::U32), ValueState::Inline, Some(Op::Mul { a: b_stride, b: tile_n }));

        kernel.raw.load_matrix_sync(frag_a, a_idx, k);
        kernel.raw.load_matrix_sync(frag_b, b_idx, n);

        Ok(())
    })?;

    kernel.raw.store_matrix_sync(frag_c, c_idx, n);

    // deepest.extend(a_deepest);
    // deepest.extend(b_deepest);

    Ok(deepest)
}
