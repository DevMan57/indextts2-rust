use indextts2::text::{TextNormalizer, TextTokenizer};

fn main() {
    let normalizer = TextNormalizer::new(false);
    let tokenizer = TextTokenizer::load("checkpoints/tokenizer_english.json", normalizer).unwrap();
    
    let text = "Hello";
    let ids = tokenizer.encode(text).unwrap();
    let tokens = tokenizer.convert_ids_to_tokens(&ids);
    
    println!("Text: {}", text);
    println!("IDs: {:?}", ids);
    println!("Tokens: {:?}", tokens);
}
