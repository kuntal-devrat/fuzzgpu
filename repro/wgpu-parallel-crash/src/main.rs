//! Minimal reproducer for a wgpu crash under concurrent compute dispatch on
//! Intel Iris Xe (Tiger Lake iGPU).
//!
//! The original finding (in the fuzzgpu test suite, wgpu 24.0.5):
//!   - 3+ threads concurrently dispatching compute on a *shared* device crash
//!     the whole process with heap corruption on DX12 (0xC0000374) or a
//!     segfault on Vulkan.
//!   - 2 concurrent dispatchers are stable (300/300 process runs passed).
//!   - Serial execution is stable (300/300 passed).
//!   - Crash rate is roughly 1 in 40-230 process runs.
//!
//! This crate mirrors the failing workload as closely as a minimal program
//! can: the REAL Levenshtein kernel WGSL from fuzzgpu, the real buffer packing
//! (input buffers via `create_buffer_init`, i.e. mapped-at-creation), a
//! MAP_READ staging readback, and N threads concurrently dispatching on one
//! shared device, each in its own submit -> poll -> map -> read -> drop loop.
//!
//! Usage:
//!   cargo run --release -- --threads 6 --iters 100            # in-process loop
//!   cargo run --release -- --process-loop 150 --threads 6     # spawns children
//!   cargo run --release -- --threads 6 --recreate-pipeline    # churn pipelines
//!
//! Backend selection: WGPU_BACKEND=dx12 (default on Windows) or =vulkan.

use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;

// The real fuzzgpu Levenshtein batch kernel, untouched.
const SHADER: &str = include_str!("shaders/levenshtein.wgsl");

const BATCH: usize = 1000; // pairs per dispatch, like the real batch tests
const MAX_LEN: usize = 32; // chars per string

struct Ctx {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
}

fn make_pipeline(device: &wgpu::Device, layout: &wgpu::PipelineLayout) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("levenshtein shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("levenshtein pipeline"),
        layout: Some(layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// Pack BATCH string pairs exactly like fuzzgpu's `compute_gpu_subset`:
/// offsets + chars vectors, u32-per-char. Pair 0 is ("a", "a") so the first
/// result word is a deterministic 0 for the readback assert.
fn pack_pairs(seed: u64) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let mut make_str = || {
        let len = (state >> 33) as usize % (MAX_LEN + 1);
        let mut s = String::new();
        for _ in 0..len {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            s.push((b'a' + ((state >> 33) as u8 % 26)) as char);
        }
        s
    };

    let mut offsets_a = vec![0u32];
    let mut chars_a = Vec::new();
    let mut offsets_b = vec![0u32];
    let mut chars_b = Vec::new();
    for i in 0..BATCH {
        let (a, b) = if i == 0 { ("a".to_string(), "a".to_string()) } else { (make_str(), make_str()) };
        chars_a.extend(a.chars().map(|c| c as u32));
        offsets_a.push(chars_a.len() as u32);
        chars_b.extend(b.chars().map(|c| c as u32));
        offsets_b.push(chars_b.len() as u32);
    }
    (offsets_a, chars_a, offsets_b, chars_b)
}

fn worker(ctx: Arc<Ctx>, id: usize, iters: usize, recreate_pipeline: bool) {
    let mut pipeline = ctx.pipeline.clone();
    for i in 0..iters {
        if recreate_pipeline && i > 0 && i % 50 == 0 {
            pipeline = make_pipeline(&ctx.device, &ctx.pipeline_layout);
        }

        let (offsets_a, chars_a, offsets_b, chars_b) = pack_pairs(0x1234_5678 ^ (id as u64) << 32 ^ i as u64);

        // Input buffers: created mapped-at-creation and filled, exactly like
        // the real kernels' `create_buffer_init` calls.
        let buf_offsets_a = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("oa"), contents: bytemuck_cast(&offsets_a), usage: wgpu::BufferUsages::STORAGE,
        });
        let buf_chars_a = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ca"), contents: bytemuck_cast(&chars_a), usage: wgpu::BufferUsages::STORAGE,
        });
        let buf_offsets_b = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ob"), contents: bytemuck_cast(&offsets_b), usage: wgpu::BufferUsages::STORAGE,
        });
        let buf_chars_b = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cb"), contents: bytemuck_cast(&chars_b), usage: wgpu::BufferUsages::STORAGE,
        });
        let params_data: [u32; 3] = [BATCH as u32, MAX_LEN as u32, 0];
        let buf_params = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"), contents: bytemuck_cast(&params_data), usage: wgpu::BufferUsages::UNIFORM,
        });

        // Output + staging: plain create_buffer, like the real readback path.
        let out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out"),
            size: (BATCH as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &ctx.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_offsets_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_chars_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf_offsets_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: buf_chars_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: buf_params.as_entire_binding() },
            ],
        });

        let mut encoder =
            ctx.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("enc") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // NOTE: renamed to `dispatch_workgroups` in wgpu v30.
            pass.dispatch_workgroups((BATCH as u32 + 63) / 64, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out, 0, &staging, 0, 4);
        ctx.queue.submit([encoder.finish()]);

        // The readback pattern from the failing test suite: register map_async,
        // poll the shared device, block on the callback channel.
        let (tx, rx) = mpsc::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        ctx.device.poll(wgpu::Maintain::Wait);
        match rx.recv_timeout(Duration::from_secs(10)) {
            Ok(Ok(())) => {
                let bytes = staging.slice(..).get_mapped_range();
                let value = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                assert_eq!(value, 0, "unexpected readback value (pair 0 is 'a' vs 'a')");
                drop(bytes);
                staging.unmap();
            }
            other => panic!("readback failed: {:?}", other),
        }

        drop(bind_group);
        drop(staging);
        drop(out);
        drop(buf_params);
        drop(buf_chars_b);
        drop(buf_offsets_b);
        drop(buf_chars_a);
        drop(buf_offsets_a);

        if i % 100 == 0 {
            println!("thread {id}: iter {i}");
            let _ = std::io::Write::flush(&mut std::io::stdout());
        }
    }
    println!("thread {id}: done");
}

fn bytemuck_cast<T: bytemuck::Pod>(v: &[T]) -> &[u8] {
    bytemuck::cast_slice(v)
}

fn run_loop(threads: usize, iters: usize, recreate_pipeline: bool) {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .unwrap_or_else(|| {
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::None,
            compatible_surface: None,
            force_fallback_adapter: true,
        }))
        .expect("no adapter")
    });
    let info = adapter.get_info();
    println!("adapter: {} ({:?})", info.name, info.backend);
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
            .expect("device");

    let bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                bg_entry(0, wgpu::BufferBindingType::Storage { read_only: true }),
                bg_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                bg_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                bg_entry(3, wgpu::BufferBindingType::Storage { read_only: true }),
                bg_entry(4, wgpu::BufferBindingType::Storage { read_only: false }),
                bg_entry(5, wgpu::BufferBindingType::Uniform),
            ],
        });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = make_pipeline(&device, &pipeline_layout);

    let ctx = Arc::new(Ctx {
        device: Arc::new(device),
        queue: Arc::new(queue),
        pipeline,
        bind_group_layout,
        pipeline_layout,
    });

    let start = Instant::now();
    let handles: Vec<_> = (0..threads)
        .map(|id| {
            let ctx = Arc::clone(&ctx);
            std::thread::spawn(move || worker(ctx, id, iters, recreate_pipeline))
        })
        .collect();
    for h in handles {
        h.join().expect("worker panicked");
    }
    println!(
        "OK: {iters} iters x {threads} threads completed in {:?} without crashing",
        start.elapsed()
    );
}

fn bg_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |name: &str, default: usize| -> usize {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(default)
    };
    let threads = get("--threads", 6);
    let iters = get("--iters", 300);
    let recreate_pipeline = args.iter().any(|a| a == "--recreate-pipeline");

    if let Some(pos) = args.iter().position(|a| a == "--process-loop") {
        // Parent mode: spawn `--child` processes and fail on the first crash,
        // mirroring the original test-harness loop that found the crash.
        let runs: usize = args
            .get(pos + 1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(300);
        let exe = std::env::current_exe().expect("current exe");
        for run in 1..=runs {
            print!("run {run}/{runs}: ");
            let _ = std::io::Write::flush(&mut std::io::stdout());
            let status = std::process::Command::new(&exe)
                .arg("--child")
                .arg("--threads")
                .arg(threads.to_string())
                .arg("--iters")
                .arg(iters.to_string())
                .env("RUST_BACKTRACE", "1")
                .status()
                .expect("spawn child");
            if !status.success() {
                eprintln!(
                    "\nCRASH at run {run}/{runs} — exit code {:?}",
                    status.code()
                );
                std::process::exit(1);
            }
            println!("ok");
        }
        println!("all {runs} runs passed");
        return;
    }

    if args.iter().any(|a| a == "--child") {
        run_loop(threads, iters, recreate_pipeline);
        return;
    }

    run_loop(threads, iters, recreate_pipeline);
}
