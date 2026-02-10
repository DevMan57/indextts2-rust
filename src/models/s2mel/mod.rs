//! S2Mel (Semantic-to-Mel) synthesis module
//!
//! Converts semantic mel codes to continuous mel spectrograms using:
//! - Length regulator for duration/alignment
//! - DiT (Diffusion Transformer) for denoising
//! - Flow matching for iterative refinement

mod length_regulator;
mod dit;
mod flow_matching;
mod weights;

pub use length_regulator::{LengthRegulator, LengthRegulatorConfig};
pub use dit::{DiffusionTransformer, DiffusionTransformerConfig};
pub use flow_matching::{FlowMatching, FlowMatchingConfig};
pub use weights::{load_s2mel_safetensors, DiTWeights, AdaLayerNormWeights, LengthRegulatorWeights, GptLayerWeights};
