# `irforge`

Advanced graph-based GPU compiler for linear algebra and AI/ML/DL.

## Usage

```rust
use irforge::{
    dispatch::{
        CompilationOptions, SimpleDType, DebugCompilationOptions, GpuContext, Graph, MetaBuilder,
        OptCompilationOptions, RawMetaBuf,
    },
    io::BpatHeader,
    nn::{MEAN_SQUARED_ERROR, Optim, add, input, matmul},
};

const ITERS: usize = 16;

const M: u32 = 768;
const N: u32 = 384;
const K: u32 = 512;
const H: u32 = 256;

const A_VAL: f32 = 0.03;
const B_VAL: f32 = 0.02;
const C_VAL: f32 = 0.01;
const D_VAL: f32 = 0.05;
const E_VAL: f32 = 1.0;

let mut meta = MetaBuilder::new();
let m = meta.new_field();
let n = meta.new_field();
let k = meta.new_field();
let h = meta.new_field();

let optim = Optim::sgd();

let mut graph = Graph::new(MEAN_SQUARED_ERROR, optim.lower);

let ctx = GpuContext::new().unwrap();
let options = CompilationOptions {
    target: ctx.detect_target(),
    opt: OptCompilationOptions::default(),
    debug: DebugCompilationOptions::empty(),
};

graph.set_options(options);
graph.define(|| {
    let a = input(&[m, k], SimpleDType::F32);
    let b = input(&[k, n], SimpleDType::F32);
    let c = input(&[h, m], SimpleDType::F32);
    let d = input(&[n, h], SimpleDType::F32);
    let e = input(&[h, h], SimpleDType::F32);

    let x = matmul(a, b);
    let y = matmul(c, x);
    let z = matmul(y, d);
    add(z, e);
});

let meta_binding = RawMetaBuf { lr: 1e-3, dims: &[M, N, K, H] }.to_binding();
let meta = meta.build(&meta_binding).unwrap();

let saved = graph.compute_saved_nodes();

graph.topo_sort().unwrap();
graph.validate(meta).unwrap();

let saved_tensors = ctx
    .alloc_tensors(&graph, &saved, &meta_binding, &optim.state)
    .unwrap();

let ir = graph.lower(meta, &saved).unwrap();
let kernels = ctx.compile(&ir).unwrap();

let in_tensors = ctx
    .load_tensors("data/v2f32_readme.bpat")
    .unwrap_or_else(|_| {
        vec![
            ctx.init_tensor_f32(&[M, K], &[A_VAL; (M * K) as usize]).unwrap(),
            ctx.init_tensor_f32(&[K, N], &[B_VAL; (K * N) as usize]).unwrap(),
            ctx.init_tensor_f32(&[H, M], &[C_VAL; (H * M) as usize]).unwrap(),
            ctx.init_tensor_f32(&[N, H], &[D_VAL; (N * H) as usize]).unwrap(),
            ctx.init_tensor_f32(&[H, H], &[E_VAL; (H * H) as usize]).unwrap(),
        ]
    });

let mut schedule = ctx
    .schedule(
        kernels,
        &meta_binding,
        &in_tensors,
        &saved_tensors,
        &[],
        &optim.state,
    )
    .unwrap();

let mut epoch = 0;

let target = {
    let arr = [2.0; (H * H) as usize];

    ctx.init_tensor_f32(&[H, H], &arr)
}.unwrap();

for _ in 0..ITERS {
    ctx.dispatch_forward(&schedule).unwrap();
    ctx.dispatch_loss(&mut schedule, &target).unwrap();
    ctx.dispatch_backward(&schedule).unwrap();
}

ctx.dispatch_optim(&mut schedule).unwrap();

ctx.sync().unwrap();

ctx.save_tensors("data/v2f32_readme.bpat", &in_tensors, BpatHeader::BpatV2f32)
    .unwrap();
```

## Custom Operations

Implementing your own operations is more advanced than in other crates. Examples of such can be found in the source code of `irforge-nn` or the README of `fused_gpu`.
