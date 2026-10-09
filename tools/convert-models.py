"""Create separately hashed experimental models; never edit the source model."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import warnings
import numpy as np
import onnx
from onnx import numpy_helper
from onnxconverter_common import float16

EXPECTED = {
    'n': 'A4DFFE5F031E476186E3EAB59BB0BC1201364DA02FD505A2949C79FD278E30E2',
    'm': '325F267E8F24C22D67179A77AFE8B93D179098E787112F64E4784A4A0AD5D7EB',
}


def sha(data):
    return hashlib.sha256(data).hexdigest().upper()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--model', choices=EXPECTED, default='m')
    args = parser.parse_args()
    expected = EXPECTED[args.model]
    raw = args.source.read_bytes()
    if sha(raw) != expected:
        raise ValueError(f'Original {args.model} model hash does not match the formal model contract')
    args.output.mkdir(parents=True, exist_ok=False)
    original = onnx.load_model_from_string(raw)
    onnx.checker.check_model(original)
    # Keep the converter defaults explicit; record every changed/clipped constant.
    minimum, maximum = 1e-7, 1e4
    arrays = [numpy_helper.to_array(t) for t in original.graph.initializer if t.data_type == onnx.TensorProto.FLOAT]
    for node in original.graph.node:
        arrays.extend(numpy_helper.to_array(a.t) for a in node.attribute
                      if a.type == onnx.AttributeProto.TENSOR and a.t.data_type == onnx.TensorProto.FLOAT)
    stats = {'float_constants': sum(a.size for a in arrays),
             'clip_small': sum(int(np.count_nonzero((np.abs(a) > 0) & (np.abs(a) < minimum))) for a in arrays),
             'clip_large': sum(int(np.count_nonzero(np.isfinite(a) & (np.abs(a) > maximum))) for a in arrays),
             'nonfinite': sum(int(np.count_nonzero(~np.isfinite(a))) for a in arrays)}
    if stats['nonfinite']:
        raise ValueError('Source constants contain nonfinite values')
    manifest = {'source': str(args.source.resolve()), 'source_sha256': expected, 'model': args.model,
                'versions': {p: importlib.metadata.version(p) for p in ['onnx', 'onnxconverter-common', 'numpy']},
                'ir_version': original.ir_version,
                'opsets': {o.domain: o.version for o in original.opset_import},
                'keep_io_types': True, 'min_positive_val': minimum, 'max_finite_val': maximum,
                'constant_stats_before_conversion': stats, 'models': {}}
    prefix = 'face-mosaic-lite' if args.model == 'n' else 'face-mosaic-detail'
    baseline = args.output / f'{prefix}-fp32.onnx'
    baseline.write_bytes(raw)
    manifest['models']['fp32'] = {'file': baseline.name, 'sha256': expected, 'bytes': len(raw)}
    # Keeping Concat in FP16 also avoids converter 1.16.0's graph-output
    # Cast-pair cleanup error when the final Concat is blocked. This candidate
    # is explicitly Conv+Concat mixed precision, not Conv-only precision.
    for name, blocks in [('fp16', None), ('conv-concat-mixed', sorted({n.op_type for n in original.graph.node if n.op_type not in {'Conv', 'Concat'}}))]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            converted = float16.convert_float_to_float16(
                onnx.load_model_from_string(raw), keep_io_types=True,
                min_positive_val=minimum, max_finite_val=maximum, op_block_list=blocks)
        if converted.ir_version != original.ir_version or list(converted.opset_import) != list(original.opset_import):
            raise ValueError('Conversion unexpectedly changed IR/opset')
        onnx.checker.check_model(converted, full_check=True)
        for side in [converted.graph.input, converted.graph.output]:
            if side[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
                raise ValueError('External I/O must remain float32')
        data = converted.SerializeToString()
        file = args.output / f'{prefix}-{name}.onnx'
        file.write_bytes(data)
        manifest['models'][name] = {
            'file': file.name, 'sha256': sha(data), 'bytes': len(data),
            'blocked_ops': list(float16.DEFAULT_OP_BLOCK_LIST) if blocks is None else blocks,
            'ops': dict(Counter(n.op_type for n in converted.graph.node)),
            'initializer_types': dict(Counter(onnx.TensorProto.DataType.Name(t.data_type) for t in converted.graph.initializer)),
            'warnings': [str(w.message) for w in caught]}
    if sha(args.source.read_bytes()) != expected:
        raise ValueError('Source changed during conversion')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'constant_stats': stats, 'models': {k: {'bytes': v['bytes'], 'sha256': v['sha256']} for k, v in manifest['models'].items()}}, indent=2))


if __name__ == '__main__':
    main()
