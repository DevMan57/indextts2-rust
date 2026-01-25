//! Benchmarks for IndexTTS2 inference pipeline
//!
//! Run with: cargo bench

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use candle_core::{Device, Tensor, DType};
use std::time::Duration;

// Import the library
use indextts2::text::{TextNormalizer, segment_text_string};
use indextts2::audio::{MelSpectrogram, Resampler};
use indextts2::models::s2mel::{LengthRegulator, DiffusionTransformer, FlowMatching};
use indextts2::models::vocoder::BigVGAN;
use indextts2::models::gpt::UnifiedVoice;
use indextts2::models::speaker::CAMPPlus;
use indextts2::inference::InferenceConfig;

/// Benchmark text normalization
fn bench_text_normalization(c: &mut Criterion) {
    let normalizer = TextNormalizer::new(false);
    let texts = vec![
        "Hello world.",
        "I have 42 apples and 123 oranges.",
        "Dr. Smith went to Washington D.C. yesterday.",
        "The quick brown fox jumps over the lazy dog. This is a longer sentence with more words to process.",
    ];

    let mut group = c.benchmark_group("text_normalization");
    for (i, text) in texts.iter().enumerate() {
        group.bench_with_input(BenchmarkId::new("normalize", i), text, |b, text| {
            b.iter(|| normalizer.normalize(black_box(*text)))
        });
    }
    group.finish();
}

/// Benchmark text segmentation
fn bench_text_segmentation(c: &mut Criterion) {
    let long_text = "Hello world. This is a test. The quick brown fox jumps over the lazy dog. \
                     How are you doing today? I hope everything is going well. \
                     This is a longer piece of text that should be segmented into multiple parts.";

    let mut group = c.benchmark_group("text_segmentation");
    for max_len in [50, 100, 200] {
        group.bench_with_input(BenchmarkId::new("segment", max_len), &max_len, |b, &max_len| {
            b.iter(|| segment_text_string(black_box(long_text), max_len))
        });
    }
    group.finish();
}

/// Benchmark mel spectrogram computation
fn bench_mel_spectrogram(c: &mut Criterion) {
    let mel = MelSpectrogram::new(1024, 256, 1024, 80, 22050, 0.0, None);

    // Generate test audio of different lengths
    let durations_sec = [0.5, 1.0, 2.0, 5.0];

    let mut group = c.benchmark_group("mel_spectrogram");
    group.measurement_time(Duration::from_secs(10));

    for duration in durations_sec {
        let samples: Vec<f32> = (0..(22050.0 * duration) as usize)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 22050.0).sin())
            .collect();

        group.bench_with_input(
            BenchmarkId::new("compute", format!("{:.1}s", duration)),
            &samples,
            |b, samples| {
                b.iter(|| mel.compute(black_box(samples)).unwrap())
            },
        );
    }
    group.finish();
}

/// Benchmark resampling
fn bench_resampler(c: &mut Criterion) {
    let sample_rates = [(48000, 22050), (44100, 16000), (16000, 22050)];

    let mut group = c.benchmark_group("resampler");
    group.measurement_time(Duration::from_secs(10));

    for (from_sr, to_sr) in sample_rates {
        // Generate 1 second of audio at source rate
        let samples: Vec<f32> = (0..from_sr)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / from_sr as f32).sin())
            .collect();

        group.bench_with_input(
            BenchmarkId::new("resample", format!("{}→{}", from_sr, to_sr)),
            &samples,
            |b, samples| {
                b.iter(|| Resampler::resample(black_box(samples), from_sr, to_sr).unwrap())
            },
        );
    }
    group.finish();
}

/// Benchmark model initialization (without weights)
fn bench_model_init(c: &mut Criterion) {
    let device = Device::Cpu;

    let mut group = c.benchmark_group("model_init");
    group.measurement_time(Duration::from_secs(5));

    group.bench_function("UnifiedVoice", |b| {
        b.iter(|| UnifiedVoice::new(black_box(&device)).unwrap())
    });

    group.bench_function("CAMPPlus", |b| {
        b.iter(|| CAMPPlus::new(black_box(&device)).unwrap())
    });

    group.bench_function("DiffusionTransformer", |b| {
        b.iter(|| DiffusionTransformer::new(black_box(&device)).unwrap())
    });

    group.bench_function("BigVGAN", |b| {
        b.iter(|| BigVGAN::new(black_box(&device)).unwrap())
    });

    group.bench_function("LengthRegulator", |b| {
        b.iter(|| LengthRegulator::new(black_box(&device)).unwrap())
    });

    group.finish();
}

/// Benchmark DiT forward pass (with random weights)
fn bench_dit_forward(c: &mut Criterion) {
    let device = Device::Cpu;
    let dit = DiffusionTransformer::new(&device).unwrap();

    let mut group = c.benchmark_group("dit_forward");
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(10);

    for seq_len in [50, 100, 200] {
        let x = Tensor::randn(0.0f32, 1.0, (1, seq_len, 80), &device).unwrap();
        let prompt_x = Tensor::zeros((1, seq_len, 80), DType::F32, &device).unwrap();
        let t = Tensor::new(&[0.5f32], &device).unwrap();
        let content = Tensor::randn(0.0f32, 1.0, (1, seq_len, 512), &device).unwrap();
        let style = Tensor::randn(0.0f32, 1.0, (1, 192), &device).unwrap();

        group.bench_with_input(
            BenchmarkId::new("forward", seq_len),
            &(x.clone(), prompt_x.clone(), t.clone(), content.clone(), style.clone()),
            |b, (x, prompt_x, t, content, style)| {
                b.iter(|| dit.forward(black_box(x), black_box(prompt_x), black_box(t), black_box(content), black_box(style)).unwrap())
            },
        );
    }
    group.finish();
}

/// Benchmark BigVGAN vocoder (with random weights)
fn bench_vocoder_forward(c: &mut Criterion) {
    let device = Device::Cpu;
    let vocoder = BigVGAN::new(&device).unwrap();

    let mut group = c.benchmark_group("vocoder_forward");
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(10);

    for seq_len in [50, 100, 200] {
        let mel = Tensor::randn(0.0f32, 1.0, (1, 80, seq_len), &device).unwrap();

        group.bench_with_input(
            BenchmarkId::new("forward", seq_len),
            &mel,
            |b, mel| {
                b.iter(|| vocoder.forward(black_box(mel)).unwrap())
            },
        );
    }
    group.finish();
}

/// Benchmark flow matching sampling
fn bench_flow_matching(c: &mut Criterion) {
    let device = Device::Cpu;
    let dit = DiffusionTransformer::new(&device).unwrap();
    let fm = FlowMatching::new(&device);

    let mut group = c.benchmark_group("flow_matching");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(10);

    let seq_len = 50; // Keep small for reasonable benchmark time

    let noise = fm.sample_noise(&[1, seq_len, 80]).unwrap();
    let prompt_x = Tensor::zeros((1, seq_len, 80), DType::F32, &device).unwrap();
    let content = Tensor::randn(0.0f32, 1.0, (1, seq_len, 512), &device).unwrap();
    let style = Tensor::randn(0.0f32, 1.0, (1, 192), &device).unwrap();

    group.bench_function("sample_25_steps", |b| {
        b.iter(|| {
            fm.sample(
                black_box(&dit),
                black_box(&noise),
                black_box(&prompt_x),
                black_box(&content),
                black_box(&style),
                0,
            ).unwrap()
        })
    });

    group.finish();
}

/// Benchmark length regulator
fn bench_length_regulator(c: &mut Criterion) {
    let device = Device::Cpu;
    let regulator = LengthRegulator::new(&device).unwrap();

    let mut group = c.benchmark_group("length_regulator");
    group.measurement_time(Duration::from_secs(10));

    for seq_len in [50, 100, 200] {
        let codes = Tensor::randn(0.0f32, 1.0, (1, seq_len, 1024), &device).unwrap();

        group.bench_with_input(
            BenchmarkId::new("forward", seq_len),
            &codes,
            |b, codes| {
                b.iter(|| regulator.forward(black_box(codes), None).unwrap())
            },
        );
    }
    group.finish();
}

/// Benchmark inference config creation
fn bench_inference_config(c: &mut Criterion) {
    c.bench_function("inference_config_default", |b| {
        b.iter(|| InferenceConfig::default())
    });
}

criterion_group!(
    benches,
    bench_text_normalization,
    bench_text_segmentation,
    bench_mel_spectrogram,
    bench_resampler,
    bench_model_init,
    bench_inference_config,
);

criterion_group!(
    name = slow_benches;
    config = Criterion::default().sample_size(10);
    targets = bench_dit_forward, bench_vocoder_forward, bench_flow_matching, bench_length_regulator
);

criterion_main!(benches, slow_benches);
