# Data Download Guide (for Teammates)

This guide explains how to download training data under AWS direct-network environments.

## Quick Start

Run from `slime/examples/aws_kiro_infra`:

```bash
bash data/download_all_data.sh --output-dir /data/r2egym_subset
```

This does two steps in order:

1. Download and preprocess dataset to:
   - `/data/r2egym_subset/train.parquet`
2. Download Docker images and save tarballs to:
   - `/data/r2egym_subset/images/*.tar.gz`

## Default Behavior

- No proxy needed (AWS direct egress to HuggingFace and Docker Hub)
- Docker image downloads are parallel
- Default parallel workers: `20`
- Step 2 runs in background via `nohup` by default
- Re-run is resumable: existing tarballs are skipped

## Common Variants

Smoke test (`10` samples, run in foreground):

```bash
bash data/download_all_data.sh --output-dir /tmp/smoke --max-samples 10 --sync
```

Only download tarballs if parquet is already prepared:

```bash
bash data/download_all_data.sh --output-dir /data/r2egym_subset --skip-parquet
```

Increase parallel download workers:

```bash
bash data/download_all_data.sh --output-dir /data/r2egym_subset --parallel 8
```

## Monitor Progress

```bash
tail -f /data/r2egym_subset/download.log
watch -n 30 'ls /data/r2egym_subset/images/*.tar.gz 2>/dev/null | wc -l; du -sh /data/r2egym_subset/images'
```

## Can downloaded tarballs be used with `docker run` later?

Yes.

Tarballs are saved from `docker save`, so they can be loaded back and run later:

```bash
# load a tarball (gzip is supported by docker load on modern Docker)
docker load -i /data/r2egym_subset/images/<instance_id>.tar.gz

# if your docker version does not support .tar.gz directly:
gunzip -c /data/r2egym_subset/images/<instance_id>.tar.gz | docker load
```

After loading, run by image name printed in `docker load` output.
If needed, verify loaded images:

```bash
docker image ls | rg 'sweb\.eval\.x86_64|r2e'
```

## Notes

- Keep enough disk space before full download (can be very large).
- `--output-dir` controls all output locations:
  - `<output-dir>/train.parquet`
  - `<output-dir>/images/`
  - `<output-dir>/download.log`
