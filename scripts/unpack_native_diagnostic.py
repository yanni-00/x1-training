"""Extract only verified CSV/JSON from our native diagnostic PT bundle."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    saved = torch.load(args.bundle, map_location='cpu', weights_only=True)
    if saved.get('format') != 'beyondmimic_native_diagnostic_v1':
        raise ValueError('Not a native diagnostic bundle; do not use policy checkpoints here')
    output = args.output.resolve()
    pending = []
    for name, record in saved['files'].items():
        path = (output / name).resolve()
        path.relative_to(output)
        if Path(name).is_absolute() or path.suffix not in ['.csv', '.json'] or path.exists():
            raise ValueError('Unsafe, unexpected or existing output: ' + name)
        tensor = record['bytes']
        if tensor.dtype != torch.uint8 or tensor.ndim != 1:
            raise ValueError('Invalid diagnostic byte tensor: ' + name)
        content = tensor.numpy().tobytes()
        if hashlib.sha256(content).hexdigest() != record['sha256']:
            raise ValueError('Diagnostic checksum mismatch: ' + name)
        pending.append((path, content))
    # Check every entry before writing any artifact.
    for path, content in pending:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    print(json.dumps({'files_extracted': len(pending), 'summary': saved['summary']}, indent=2))


if __name__ == '__main__':
    main()
