use irforge::{
    dispatch::{
        CompilationOptions, DebugCompilationOptions, GpuContext, Graph, MetaBuilder,
        OptCompilationOptions, SimpleDType,
    },
    nn::{MEAN_SQUARED_ERROR, Optim, add, input, mul},
    tensor::f16,
};

#[test]
fn mul_add_f16() {
    let mut meta = MetaBuilder::new();
    let m = meta.new_field();
    let n = meta.new_field();

    let state = Optim::sgd().state;

    let mut graph = Graph::new(MEAN_SQUARED_ERROR, Optim::sgd().lower);

    graph.define(|| {
        let a = input(&[m, n], SimpleDType::F16);
        let b = input(&[m, n], SimpleDType::F16);
        let c = input(&[m, n], SimpleDType::F16);

        let x = mul(a, b);
        add(c, x);
    });

    let meta_binding = [1e-3_f32.to_bits(), 32, 32];
    let meta = meta.build(&meta_binding).unwrap();

    let saved = graph.compute_saved_nodes();

    graph.topo_sort().unwrap();
    graph.validate(meta).unwrap();

    let Ok(ctx) = GpuContext::new() else {
        return;
    };
    let options = CompilationOptions {
        target: ctx.detect_target(),
        opt: OptCompilationOptions::default(),
        debug: DebugCompilationOptions::empty(),
    };

    let meta_binding = ctx.alloc_meta(&meta_binding);
    let saved_tensors = ctx
        .alloc_tensors(&graph, &saved, meta_binding, &state)
        .unwrap();

    let ir = graph.lower(meta, &options, &saved).unwrap();
    let kernels = ctx.compile(&ir, &options).unwrap();

    let in_tensors = [
        ctx.init_tensor_f16(&[32, 32], &[f16::from_f32(3.0); 1024])
            .unwrap(),
        ctx.init_tensor_f16(&[32, 32], &[f16::from_f32(2.0); 1024])
            .unwrap(),
        ctx.init_tensor_f16(&[32, 32], &[f16::from_f32(1.0); 1024])
            .unwrap(),
    ];

    ctx.upload(&saved_tensors.seed, &[f16::from_f32(1.0); 1024], 0)
        .unwrap();

    let schedule = ctx
        .schedule(
            kernels,
            meta_binding,
            &in_tensors,
            &saved_tensors,
            &[],
            &state,
        )
        .unwrap();

    ctx.dispatch_forward(&schedule).unwrap();
    ctx.dispatch_backward(&schedule).unwrap();

    ctx.sync().unwrap();

    let mut dst = [f16::from_f32(0_f32); 1024];

    let out_tensor = &saved_tensors.forward_out;
    let grad_tensors = &saved_tensors.grad_tensors;

    ctx.download(out_tensor, &mut dst, 0).unwrap();
    assert!(dst.iter().all(|x| *x == f16::from_f32(7.0)));

    ctx.download(&grad_tensors[0], &mut dst, 0).unwrap();
    assert!(dst.iter().all(|x| *x == f16::from_f32(2.0)));

    ctx.download(&grad_tensors[1], &mut dst, 0).unwrap();
    assert!(dst.iter().all(|x| *x == f16::from_f32(3.0)));

    ctx.download(&grad_tensors[2], &mut dst, 0).unwrap();
    assert!(dst.iter().all(|x| *x == f16::from_f32(1.0)));
}
