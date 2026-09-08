# vLLM Apptainer image

The benchmark uses the official vLLM OpenAI image for vLLM 0.15.1, converted
to one immutable Apptainer SIF. The amd64 OCI manifest is pinned rather than
resolved through a mutable tag:

```text
docker://vllm/vllm-openai@sha256:06f9f0d5c7cb079504615c51dab70cd18abbf609d1358b940172181ac0a92efa
```

Docker Hub reported this digest for the amd64 variant of
`vllm/vllm-openai:v0.15.1` on 2026-09-08. The compressed OCI image is about
9.13 GB. Building requires substantially more temporary space while layers are
expanded.

Build from the repository root. Temporary OCI files stay on node-local `/tmp`;
the output directory receives only the final SIF:

```bash
scripts/build_vllm_apptainer.sh \
    containers/vllm-openai-v0.15.1-amd64.sif \
    /tmp
```

The SIF and writable overlay files in this directory are intentionally ignored
by Git. Do not use a sandbox container because it would recreate the
many-small-file problem on shared storage.

The base image is read-only. A separate per-job EXT3 overlay will hold the
temporary MoE timing patch. Run with `apptainer exec --nv` on the allocated
`uranus` or `neptune` compute node; the host NVIDIA driver must not be installed
inside the image.

The official image uses `python3` rather than a `python` command. Its observed
offline versions are Python 3.12.12, PyTorch 2.9.1+cu129, vLLM 0.15.1, and
aiohttp 3.13.3. It also contains `/usr/bin/patch`. The timing-patch manager
recognizes the image's vLLM source as the exact unpatched state.

Verify GPU passthrough on the required compute node from the repository root:

```bash
sbatch --account=YOUR_MINERVA_ACCOUNT scripts/verify_vllm_apptainer.sbatch
```

This creates a unique evidence directory under `results/raw/` and does not
start a vLLM server or modify the image. Do not submit it on the Vera cluster:
the repository safety contract permits only the Minerva nodes named `uranus`
or `neptune`.
