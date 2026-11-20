"""check_env.py - memeriksa GPU, PyTorch, dan paket Hugging Face sebelum pelatihan panjang."""
import importlib
import sys

import common as C


def baris(ok, msg):
    print(("[OK]    " if ok else "[GAGAL] ") + msg)
    return ok


def main():
    semua = True
    baris(True, f"Python {sys.version.split()[0]}")
    for mod in ("numpy", "pandas", "scipy", "sklearn", "transformers", "sentencepiece", "google.protobuf"):
        try:
            m = importlib.import_module(mod)
            baris(True, f"{mod} {getattr(m, '__version__', '')}")
        except Exception as e:
            semua &= baris(False, f"{mod} tidak bisa diimpor: {e}")
    try:
        import torch
    except Exception as e:
        raise SystemExit(f"[GAGAL] torch tidak terpasang: {e}\nLihat README bagian 1 (instal PyTorch cu128).")
    baris(True, f"torch {torch.__version__} (CUDA build: {torch.version.cuda})")
    if not torch.cuda.is_available():
        semua &= baris(False, "GPU tidak terdeteksi oleh PyTorch. Periksa driver NVIDIA dan versi torch (RTX 50xx butuh build cu128 atau lebih baru).")
    else:
        cap = torch.cuda.get_device_capability(0)
        baris(True, f"GPU: {torch.cuda.get_device_name(0)}, compute capability {cap[0]}.{cap[1]}, memori {torch.cuda.get_device_properties(0).total_memory / 2 ** 30:.1f} GB")
        sm = f"sm_{cap[0]}{cap[1]}"
        semua &= baris(any(x.startswith(sm) or x.startswith(f'compute_{cap[0]}{cap[1]}') for x in torch.cuda.get_arch_list()),
                       f"build torch memuat kernel {sm} (daftar: {', '.join(torch.cuda.get_arch_list())})")
        try:
            a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            (a @ a).sum().backward()
            torch.cuda.synchronize()
            baris(True, "perkalian matriks bf16 dan backward berjalan di GPU")
            baris(torch.cuda.is_bf16_supported(), "bf16 didukung")
        except Exception as e:
            semua &= baris(False, f"operasi GPU gagal: {e}")
    cfg = C.config()
    for kunci in ("model_deberta", "model_bert"):
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(cfg[kunci])
            baris(True, f"tokenizer {cfg[kunci]} terunduh dan berjalan ({len(tok('This is a test essay.')['input_ids'])} token pada kalimat uji)")
        except Exception as e:
            semua &= baris(False, f"tokenizer {cfg[kunci]} gagal dimuat: {repr(e)[:300]}\n         (pastikan sentencepiece dan protobuf terpasang dan internet ke huggingface.co terbuka)")
    print("\nLingkungan siap." if semua else "\nAda yang perlu diperbaiki sebelum lanjut.")
    sys.exit(0 if semua else 1)


if __name__ == "__main__":
    main()
