//! Semantic encoding and codec
//!
//! Wav2Vec-BERT 2.0 semantic feature extraction
//! and semantic codec for quantization

mod wav2vec_bert;
mod codec;
mod vocos;

pub use wav2vec_bert::SemanticEncoder;
pub use codec::SemanticCodec;
pub use vocos::VocosBackbone;
