import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model_forecasting():
    models = {}
    indicators = ['Rupiah','Saham dan Modal lainnya','TRANSPORTASI','MAKANAN, MINUMAN DAN TEMBAKAU','PERLENGKAPAN, PERALATAN DAN PEMELIHARAAN RUTIN RUMAH TANGGA']
    for indicator in indicators:
        models[indicator] = tf.keras.models.load_model(f"model_fix/{indicator}_model.h5")
    return models

def load_model_advisor():
    model = AutoModelForCausalLM.from_pretrained(
        "./Llama-3.2-1B-personal-finance",
        torch_dtype=torch.float16,  # Gunakan half precision
        device_map="auto",
        low_cpu_mem_usage=True,
        offload_folder="offload"  # Untuk model besar
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Optimasi model untuk inferensi
    model.eval()  # Set ke mode evaluasi
    if torch.cuda.is_available():
        model = model.cuda()
        torch.backends.cudnn.benchmark = True 
    
    tokenizer = AutoTokenizer.from_pretrained(
        f"./Llama-3.2-1B-personal-finance",
        model_max_length=512 
    )

    return model, tokenizer

