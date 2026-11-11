# Experiment ledger

- entries: 4, submissions: 0
- rendered: 2026-11-11T09:48:48+08:00 (source of truth: ledger.jsonl)

| id | time | kind | title | fields | hashes |
|---|---|---|---|---|---|
| 1 | 2026-11-11T09:48:48+08:00 | data | T1.1 V-JEPA 2 ViT-L (HF) downloaded | model.safetensors.bytes=1303947864, model.safetensors.sha256=25466aef85727d16546c6cf8c99f12fcfad9cbca8225d45f23685e2e025b786b, path=$D/weights/vjepa/vjepa2-vitl-fpc64-256, revision=b3c1679b7c34d3255ef3547f27c7b226aefab26f |  |
| 2 | 2026-11-11T09:48:48+08:00 | data | T1.1 V-JEPA 2.1 ViT-L checkpoint downloaded | bytes=5151198524, keys=encoder,predictor,opt,scaler,ema_encoder,epoch,loss,batch_size,world_size,lr, path=$D/weights/vjepa/vjepa2_1_vitl_dist_vitG_384.pt, sha256=7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6 |  |
| 3 | 2026-11-11T09:48:48+08:00 | data | T1.1 V-JEPA 2.1 ema_encoder extracted | bytes=1218821403, params_M=304.68, path=$D/weights/vjepa/vjepa2_1_vitl_384_ema_encoder.pt, sha256=e8caa8771efcfec4f5db6b6be490cd2754c7c14deef3f746157aaec697427574, tensors=302 |  |
| 4 | 2026-11-11T09:48:48+08:00 | data | T1.1 vjepa2 source snapshot and timm | commit=204698b45b3712590f06245fbfba32d3be539812, src=$D/weights/vjepa/vjepa2-src, tar.sha256=e7e9c554c67e72c2c6884d751f4731bbf25d0d203b01d4b367b5157371b13ee1, timm=1.0.15 in $D/pylib (--no-deps) |  |
