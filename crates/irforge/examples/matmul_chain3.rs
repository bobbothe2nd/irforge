use gpu_telemetry::monitor::{GpuMonitor, telemetry::Telemetry};
use irforge::{
    dispatch::{
        CompilationOptions, DebugCompilationOptions, GpuContext, Graph, MetaBuilder,
        OptCompilationOptions, Schedule, SimpleDType,
    },
    nn::{MEAN_SQUARED_ERROR, Optim, add, input, matmul},
};
use std::time::{Duration, Instant};

fn model_runtime(ctx: &GpuContext, schedule: &Schedule) {
    const ITERS: [usize; 6] = [50, 100, 250, 100, 10, 1];

    let mut prev_iters = ITERS[0];

    for _ in 0..prev_iters {
        ctx.dispatch_forward(schedule).unwrap();

        ctx.dispatch_backward(schedule).unwrap();
    }

    for iters in ITERS.into_iter().skip(1) {
        let sync_start = Instant::now();

        for _ in 0..iters {
            ctx.dispatch_forward(schedule).unwrap();
            ctx.dispatch_backward(schedule).unwrap();
        }

        let sync_elapsed = sync_start.elapsed();

        println!(" MODEL {prev_iters}: {sync_elapsed:?} elapsed");

        prev_iters = iters;
    }

    let sync_start = Instant::now();

    ctx.sync().unwrap();

    let sync_elapsed = sync_start.elapsed();

    println!(" MODEL 1: {sync_elapsed:?} elapsed");
}

fn main() {
    const M: u32 = 768;
    const N: u32 = 1024;
    const K: u32 = 512;
    const H: u32 = 256;

    const A_VAL: f32 = 3.0;
    const B_VAL: f32 = 2.0;
    const C_VAL: f32 = 1.0;
    const D_VAL: f32 = 0.5;
    const E_VAL: f32 = 1.0;

    let mut meta = MetaBuilder::new();
    let m = meta.new_field();
    let n = meta.new_field();
    let k = meta.new_field();
    let h = meta.new_field();

    let optim = Optim::sgd();

    let mut graph = Graph::new(MEAN_SQUARED_ERROR, optim.lower);

    let ctx = GpuContext::new().unwrap();
    let target = ctx.detect_target();

    graph.set_target(target);
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

    let meta_binding = [1e-3_f32.to_bits(), M, N, K, H];
    let meta = meta.build(&meta_binding).unwrap();

    let saved = graph.compute_saved_nodes();

    graph.topo_sort().unwrap();
    graph.validate(meta).unwrap();

    let options = CompilationOptions {
        target: graph.get_target().unwrap(),
        opt: OptCompilationOptions::default(),
        debug: DebugCompilationOptions::empty(),
    };

    let meta_binding = ctx.alloc_meta(&meta_binding);
    let saved_tensors = ctx
        .alloc_tensors(&graph, &saved, meta_binding, &optim.state)
        .unwrap();

    let compile_start = Instant::now();

    let ir = graph.lower(meta, &options, &saved).unwrap();
    let kernels = ctx.compile(&ir, &options).unwrap();

    let compile_elapsed = compile_start.elapsed();

    println!("COMPILE TIME: {compile_elapsed:?} elapsed");

    let tensor_start = Instant::now();

    let in_tensors = [
        ctx.init_tensor_f32(&[M, K], &[A_VAL; (M * K) as usize])
            .unwrap(),
        ctx.init_tensor_f32(&[K, N], &[B_VAL; (K * N) as usize])
            .unwrap(),
        ctx.init_tensor_f32(&[H, M], &[C_VAL; (H * M) as usize])
            .unwrap(),
        ctx.init_tensor_f32(&[N, H], &[D_VAL; (N * H) as usize])
            .unwrap(),
        ctx.init_tensor_f32(&[H, H], &[E_VAL; (H * H) as usize])
            .unwrap(),
    ];

    ctx.upload(&saved_tensors.seed, &[1_f32; (H * H) as usize], 0)
        .unwrap();

    let tensor_elapsed = tensor_start.elapsed();

    println!("TENSOR INIT TIME: {tensor_elapsed:?} elapsed");

    let schedule = ctx
        .schedule(
            kernels,
            meta_binding,
            &in_tensors,
            &saved_tensors,
            &[],
            &optim.state,
        )
        .unwrap();

    let monitor: GpuMonitor<Telemetry> = GpuMonitor::start(Duration::from_millis(10)).unwrap();

    model_runtime(&ctx, &schedule);

    let telemetry = monitor.stop().unwrap();

    let mut max_budget = 0;
    let mut accum_usage = 0;

    for sample in &telemetry.samples {
        for heap in &sample.heaps {
            if let Some(usage) = heap.usage {
                accum_usage += usage;
            }

            if let Some(budget) = heap.budget
                && budget > max_budget
            {
                max_budget = budget;
            }
        }
    }

    let avg_usage = accum_usage / (telemetry.samples.len() as u64);

    println!("\n  AVERAGE MEMORY USAGE: {}", avg_usage);
    println!(" MAXIMUM MEMORY BUDGET: {}", max_budget);
}
