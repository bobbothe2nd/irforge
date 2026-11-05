use irforge::{
    dispatch::{
        CompilationOptions, DebugCompilationOptions, GpuContext, Graph, MetaBuilder, OptCompilationOptions, RawMetaBuf, SimpleDType,
    }, nn::{MEAN_SQUARED_ERROR, Optim, add, gelu, input, matmul, mul, sub}, tensor::f16,
};

#[test]
fn mul_add_f16() {
    let mut meta = MetaBuilder::new();
    let m = meta.new_field();
    let n = meta.new_field();

    let state = Optim::sgd().state;

    let mut graph = Graph::new(MEAN_SQUARED_ERROR, Optim::sgd().lower);

    let Ok(ctx) = GpuContext::new() else {
        return;
    };

    let options = CompilationOptions {
        target: ctx.detect_target(),
        opt: OptCompilationOptions::default(),
        debug: DebugCompilationOptions::empty(),
    };

    graph.set_options(options);
    graph.define(|| {
        let a = input(&[m, n], SimpleDType::F16);
        let b = input(&[m, n], SimpleDType::F16);
        let c = input(&[m, n], SimpleDType::F16);

        let x = mul(a, b);
        add(c, x);
    });

    let meta_binding = RawMetaBuf { lr: 1e-3, dims: &[32, 32] }.to_binding();
    let meta = meta.build(&meta_binding).unwrap();

    let saved = graph.compute_saved_nodes();

    graph.topo_sort().unwrap();
    graph.validate(meta).unwrap();

    let saved_tensors = ctx
        .alloc_tensors(&graph, &saved, &meta_binding, &state)
        .unwrap();

    let ir = graph.lower(meta, &saved).unwrap();
    let kernels = ctx.compile(&ir).unwrap();

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
            &meta_binding,
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

#[test]
fn matmul_gelu_matmul_add_f16() {
    let mut meta = MetaBuilder::new();
    let m = meta.new_field();
    let n = meta.new_field();
    let k = meta.new_field();

    let state = Optim::sgd().state;

    let mut graph = Graph::new(MEAN_SQUARED_ERROR, Optim::sgd().lower);

    let Ok(ctx) = GpuContext::new() else {
        return;
    };

    let options = CompilationOptions {
        target: ctx.detect_target(),
        opt: OptCompilationOptions::default(),
        debug: DebugCompilationOptions::empty(),
    };

    graph.set_options(options);

    graph.define(|| {
        let a = input(&[m, k], SimpleDType::F16);
        let b = input(&[k, n], SimpleDType::F16);
        let c = input(&[m, n], SimpleDType::F16);

        let gelu_b = gelu(b);

        let x = matmul(a, gelu_b);

        sub(x, c);
    });

    let meta_binding = RawMetaBuf {
        lr: 1e-3,
        dims: &[32, 32, 32],
    }
    .to_binding();

    let meta = meta.build(&meta_binding).unwrap();

    let saved = graph.compute_saved_nodes();

    graph.topo_sort().unwrap();
    graph.validate(meta).unwrap();

    let saved_tensors = ctx
        .alloc_tensors(&graph, &saved, &meta_binding, &state)
        .unwrap();

    let ir = graph.lower(meta, &saved).unwrap();
    let kernels = ctx.compile(&ir).unwrap();

    let in_tensors = [
        ctx.init_tensor_f16(
            &[32, 32],
            &[f16::from_f32(3.0); 1024],
        )
        .unwrap(),
        ctx.init_tensor_f16(
            &[32, 32],
            &[f16::from_f32(2.0); 1024],
        )
        .unwrap(),
        ctx.init_tensor_f16(
            &[32, 32],
            &[f16::from_f32(1.0); 1024],
        )
        .unwrap(),
    ];

    ctx.upload(
        &saved_tensors.seed,
        &[f16::from_f32(1.0); 1024],
        0,
    )
    .unwrap();

    let schedule = ctx
        .schedule(
            kernels,
            &meta_binding,
            &in_tensors,
            &saved_tensors,
            &[],
            &state,
        )
        .unwrap();

    ctx.dispatch_forward(&schedule).unwrap();
    ctx.dispatch_backward(&schedule).unwrap();

    ctx.sync().unwrap();

    let mut dst = [f16::from_f32(0.0); 1024];

    let out_tensor = &saved_tensors.forward_out;
    let grad_tensors = &saved_tensors.grad_tensors;

    ctx.download(out_tensor, &mut dst, 0).unwrap();

    let gelu_a = 3.0 * 0.9986501;
    let gelu_b = 2.0 * 0.9772499;

    let expected_out = 32.0 * (3.0 * gelu_b + gelu_a);

    assert!(dst.iter().all(|x| {
        (x.to_f32() - expected_out).abs() < 1.0
    }));

    let gelu_prime_3 = 0.9999780;
    let expected_da = 32.0 * (0.9772499 + gelu_prime_3);

    ctx.download(&grad_tensors[0], &mut dst, 0).unwrap();

    assert!(dst.iter().all(|x| {
        (x.to_f32() - expected_da).abs() < 1.0
    }));

    let gelu_prime_2 = 1.0852318;
    let expected_db = 96.0 * gelu_prime_2;

    ctx.download(&grad_tensors[1], &mut dst, 0).unwrap();

    assert!(dst.iter().all(|x| {
        (x.to_f32() - expected_db).abs() < 1.0
    }));

    let expected_dc = 32.0 * gelu_a;

    ctx.download(&grad_tensors[2], &mut dst, 0).unwrap();

    assert!(dst.iter().all(|x| {
        (x.to_f32() - expected_dc).abs() < 1.0
    }));
}
