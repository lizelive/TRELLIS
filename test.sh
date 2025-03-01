python ./t2i.py  --ckpt_id "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
python ./t2i.py  --ckpt_id "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"  --fuse_attn_projections --compile --compile_vae  --quantization autoquant
python ./t2i.py  --ckpt_id "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"  --fuse_attn_projections --compile --compile_vae  --quantization fp8wo