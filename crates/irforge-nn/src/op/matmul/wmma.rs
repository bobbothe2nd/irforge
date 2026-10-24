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

#[allow(unused)]
pub fn forward_matmul(
    eval_node: impl Fn(
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
    root: NodeId,
    input: NodeId,
    resolved: &mut Vec<NodeId>,
    m: ValueId,
    n: ValueId,
    k: ValueId,
    swap_a: bool,
    swap_b: bool,
    a_node: &NodeInput,
    b_node: &NodeInput,
    graph: &Graph,
    out: ValueId,
    node_params: &[Option<ParamId>],
    saved_params: &[Option<ParamId>],
    kernel: &mut LinkedKernel,
    base: ValueId,
    row: ValueId,
    col: ValueId,
    local_row: ValueId,
    local_col: ValueId,
    shared_size: u32,
    tile_size: ValueId,
    params: &[Param],
    stable_iteration_space: &mut bool,
    options: &CompilationOptions,
) -> Result<Vec<NodeId>, Error> {
    let subgroup_size = options.target.subgroup_size.ok_or(Error {
        msg: "attempted WMMA matmul with unknown subgroup size",
        kind: ErrorKind::InternalError,
        ctx: (),
    })?.get() as u32;
    let block = [subgroup_size / 8, 8, 1];
    kernel.raw.block.dim3 = block;

    let mut deepest = Vec::new();

    let zero_u = kernel.raw.def_var(
        DType::Simple(SimpleDType::U32),
        ValueState::Inline,
        Some(Op::ConstU32 { val: 0 }),
    );
    let zero_f = kernel.raw.def_var(
        DType::Simple(dtype),
        ValueState::Inline,
        Some(Op::ConstF32 { val: 0.0 }),
    );

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
        DType::MmaA { dtype, m: shared_size, n: shared_size, k: shared_size },
        ValueState::Mut,
        None,
    );
    let frag_b = kernel.raw.def_var(
        DType::MmaB { dtype, m: shared_size, n: shared_size, k: shared_size },
        ValueState::Mut,
        None,
    );
    let frag_c = kernel.raw.def_var(
        DType::MmaAccum { dtype, m: shared_size, n: shared_size, k: shared_size },
        ValueState::Mut,
        None,
    );

    kernel.raw.fill_frag(frag_c, zero_f);

    let tk = kernel.raw.def_var(
        DType::Simple(SimpleDType::F32),
        ValueState::Mut,
        Some(Op::ConstU32 { val: 0 }),
    );

    let mut a_deepest = Vec::new();
    let mut b_deepest = Vec::new();

    kernel.push_for_loop(tk, k, tile_size, |kernel| {
        let a_idx = kernel.raw.def_var(DType::Simple(SimpleDType::U32), ValueState::Inline, Some(Op::Mul { a: (), b: () }));

        kernel.raw.load_matrix_sync(frag_a, a_idx, k);

        Ok(())
    })?;

    deepest.append(&mut a_deepest);
    deepest.append(&mut b_deepest);

    Ok(deepest)
}
