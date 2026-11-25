"""AOT-compile the site's pinned ONNX17 single-sample policies to ES modules + LE f32 weights.

Shape/index work runs here, not in the worker. Scalar dataflow is emitted into
fixed-size blocks; dense and normalization kernels use contiguous typed storage.
This is intentionally a compiler for the deployed subset, not a general ONNX loader.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

F32 = np.dtype('float32')
OPS = {
    'Add': '({0}+{1})', 'Sub': '({0}-({1}))', 'Mul': '({0}*{1})', 'Div': '({0}/{1})',
    'Neg': '(-{0})', 'Cos': 'Math.cos({0})', 'Sin': 'Math.sin({0})',
    'Atan': 'Math.atan({0})', 'Sqrt': 'Math.sqrt({0})', 'Erf': 'erf({0})',
    'Less': '({0}<{1})', 'Greater': '({0}>{1})', 'Equal': '({0}==={1})',
    'Where': '({0}?{1}:{2})', 'Clip': 'Math.min(Math.max({0},{1}),{2})',
}
STATIC = {'Constant', 'ConstantOfShape', 'Add', 'Sub', 'Mul', 'Div', 'Equal', 'Where',
          'Clip', 'Less', 'Greater', 'Neg', 'Cos', 'Sin', 'Atan', 'Sqrt', 'Erf'}


def require(cond, detail):
    if not cond:
        raise ValueError(detail)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def js(x):
    if isinstance(x, str):
        return x
    if isinstance(x, (np.bool_, bool)):
        return str(int(x))
    if math.isnan(x):
        return 'NaN'
    if math.isinf(x):
        return '-Infinity' if x < 0 else 'Infinity'
    return repr(x.item() if isinstance(x, np.generic) else x)


def tensor_shape(value):
    t = value.type.tensor_type
    return t.elem_type, [d.dim_value for d in t.shape.dim]


class Compiler:
    def __init__(self, width):
        self.env = {'obs': np.array([f'x[{i}]' for i in range(width)], object).reshape(1, -1)}
        self.code = []
        self.scratch = self.weight_size = 0
        self.weights = []
        self.memo = {}
        self.weight_offsets = {}

    def reserve(self, n):
        base = self.scratch
        self.scratch += n
        return base

    def scalar(self, expr):
        if expr not in self.memo:
            idx = self.reserve(1)
            self.memo[expr] = f's[{idx}]'
            self.code.append(f's[{idx}]={expr};')
        return self.memo[expr]

    def ref(self, base, shape):
        return np.array([f's[{i}]' for i in range(base, base+math.prod(shape))], object).reshape(shape)

    def pack(self, data):
        require(data.dtype == F32 and np.isfinite(data).all(), 'Expected finite float32 parameters')
        blob = data.astype('<f4', copy=False).tobytes()
        if blob not in self.weight_offsets:
            self.weight_offsets[blob] = self.weight_size
            self.weight_size += data.size
            self.weights.append(blob)
        return self.weight_offsets[blob]

    def contiguous(self, x):
        first = x.flat[0]
        if isinstance(first, str) and first.startswith('s[') and first.endswith(']') and first[2:-1].isdigit():
            base = int(first[2:-1])
            if all(v == f's[{base+i}]' for i, v in enumerate(x.flat)):
                return base
        base = self.reserve(x.size)
        self.code.extend(f's[{base+i}]={js(v)};' for i, v in enumerate(x.flat))
        return base

    def compile_node(self, node):
        op = node.op_type
        args = [self.env[n] if n else None for n in node.input]
        a = {p.name: helper.get_attribute_value(p) for p in node.attribute}
        x = args[0] if args else None
        shape = () if x is None else x.shape
        require(not node.domain and len(node.output) == 1, 'Expected a single standard ONNX output')
        if op in STATIC and all(t is None or t.dtype != object for t in args):
            # Reference evaluation is only for compile-time constants; no Python evaluator ships.
            return ReferenceEvaluator(node, opsets={'': 17}).run(None, dict(zip(node.input, args)))[0]
        if op == 'Identity':
            return x
        if op == 'Shape':
            require(not a, 'Shape attributes')
            return np.array(shape, np.int64)
        if op == 'Slice':
            require(len(args) in (4, 5), 'Slice signature')
            _, starts, ends, axes, *steps = args
            ix = [slice(None)]*x.ndim
            for axis, start, end, step in zip(axes, starts, ends, steps[0] if steps else [1]*len(axes), strict=True):
                ix[axis] = slice(int(start), int(end), int(step))
            return x[tuple(ix)]
        if op == 'Reshape':
            require(a.get('allowzero', 0) == 0 and 0 not in args[1], 'Unsupported Reshape')
            return x.reshape(tuple(args[1]))
        if op == 'Flatten':
            axis = a['axis'] % x.ndim
            return x.reshape(math.prod(shape[:axis]), math.prod(shape[axis:]))
        if op == 'Unsqueeze':
            return np.expand_dims(x, tuple(args[1]))
        if op == 'Expand':
            return np.broadcast_to(x, np.broadcast_shapes(shape, tuple(args[1])))
        if op == 'Concat':
            return np.concatenate([t.astype(object) for t in args] if any(t.dtype == object for t in args) else args, axis=a['axis'])
        if op == 'ScatterND':
            require(not a and args[1].dtype == np.int64, 'ScatterND needs fixed indices')
            out = x.copy()
            for i in np.ndindex(args[1].shape[:-1]):
                out[tuple(args[1][i])] = args[2][i]
            return out
        if op == 'Gather':
            idx = args[1]
            if idx.dtype != object:
                return np.asarray(np.take(x, idx, axis=a['axis']))
            require(a == {'axis': 0} and x.dtype == F32 and shape == (3, 3) and idx.size == 1,
                    'Dynamic Gather is only supported for the three-mode mask')
            base = self.pack(x)
            return np.array([self.scalar(f'w[{base+i}+3*{idx.flat[0]}]') for i in range(3)], object).reshape(*idx.shape, 3)
        if op == 'Cast':
            t = a['to']
            require(t in (TensorProto.BOOL, TensorProto.INT64), 'Unsupported Cast')
            if x.dtype != object:
                return x.astype(bool if t == TensorProto.BOOL else np.int64)
            fmt = '(!!{0}?1:0)' if t == TensorProto.BOOL else 'Math.trunc({0})'
            return np.array([self.scalar(fmt.format(js(v))) for v in x.flat], object).reshape(shape)
        if op in ('ReduceSum', 'ReduceL2'):
            axes = args[1] if len(args) > 1 else a['axes']
            require(tuple(axes) in ((-1,), (len(shape)-1,)) and a.get('keepdims', 1) == 1, 'Unsupported reduction')
            rows = []
            for row in x.reshape(-1, shape[-1]):
                terms = [f'f({js(v)}*{js(v)})' if op == 'ReduceL2' else js(v) for v in row]
                total = terms[0]
                for term in terms[1:]:
                    total = f'f({total}+{term})'
                rows.append(self.scalar(f'Math.sqrt({total})' if op == 'ReduceL2' else total))
            return np.array(rows, object).reshape(*shape[:-1], 1)
        if op in OPS:
            require(not a, f'Unexpected {op} attributes')
            if op == 'Clip':
                args += [None]*(3-len(args))
                args = [args[0], np.array(-np.inf) if args[1] is None else args[1], np.array(np.inf) if args[2] is None else args[2]]
            ts = np.broadcast_arrays(*args)
            if op == 'Where':
                out = [b if bool(c) else d for c,b,d in zip(*(t.flat for t in ts))] if ts[0].dtype != object else [self.scalar(OPS[op].format(*map(js,v))) for v in zip(*(t.flat for t in ts))]
            elif op in ('Less', 'Greater', 'Equal'):
                # Store predicate results as numeric 0/1 in Float32 scratch.
                out = [self.scalar(OPS[op].format(*map(js,v))+'?1:0') for v in zip(*(t.flat for t in ts))]
            else:
                out = [self.scalar(OPS[op].format(*map(js,v))) for v in zip(*(t.flat for t in ts))]
            return np.array(out, object).reshape(ts[0].shape)
        if op == 'Gemm':
            _, w, b = args
            require(a == {'alpha': 1., 'beta': 1., 'transB': 1} and len(shape) == 2 and shape[0] == 1 and w.shape == (b.size, shape[1]) and b.shape == (b.size,), 'Unsupported Gemm ABI')
            src, dst = self.contiguous(x), self.reserve(b.size)
            self.code.append(f'dense(s,w,{src},{dst},{self.pack(w)},{self.pack(b)},{shape[1]},{b.size});')
            return self.ref(dst, (1,b.size))
        if op == 'LayerNormalization':
            _, w, b = args
            require(set(a) == {'axis','epsilon'} and a['axis'] == -1 and shape == (1,w.size) and w.shape == b.shape == (w.size,), 'Unsupported LayerNormalization ABI')
            src, dst = self.contiguous(x), self.reserve(w.size)
            self.code.append(f'norm(s,w,{src},{dst},{self.pack(w)},{self.pack(b)},{w.size},{a["epsilon"]});')
            return self.ref(dst, shape)
        raise ValueError(f'Unsupported op: {op}')

    def compile(self, model):
        self.env.update({i.name: numpy_helper.to_array(i) for i in model.graph.initializer})
        for node in model.graph.node:
            try:
                self.env[node.output[0]] = self.compile_node(node)
            except Exception as exc:
                raise ValueError(f'Failed to compile {node.op_type}: {node.name}') from exc
        for i,v in enumerate(self.env['q_des'].flat):
            self.code.append(f'y[{i}]={js(v)};')
        # Small, static functions avoid constructing runtime tensor/shape objects
        # and large monolithic functions that JITs may refuse to optimize.
        chunks = [self.code[i:i+180] for i in range(0,len(self.code),180)]
        code = "// Generated by export_onnx_to_jsbin.py; do not edit.\nimport {dense,norm,erf} from '../../js/mlp_infer.mjs?v=js-f32-v1';\nconst f=Math.fround;\n"
        for i,lines in enumerate(chunks):
            code += f'function block{i}(x,y,s,w){{\n'+'\n'.join(lines)+'\n}\n'
        code += 'export function forward(x,y,s,w){\n'+''.join(f'block{i}(x,y,s,w);\n' for i in range(len(chunks)))+'}\n'
        return code.encode(), b''.join(self.weights)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('site', type=Path)
    args = p.parse_args()
    site = args.site
    manifest = site/'assets/BUNDLE.json'
    bundle = json.loads(manifest.read_text())
    compiled = site/'models/js'
    compiled.mkdir(exist_ok=True)
    for name, entry in bundle['checkpoints'].items():
        raw = (site/entry['path']).read_bytes()
        require(digest(raw) == entry['sha256'], f'Original ONNX hash mismatch: {name}')
        model = onnx.load_model_from_string(raw)
        require(not any(i.data_location for i in model.graph.initializer), 'External ONNX data is not supported')
        onnx.checker.check_model(model)
        width = 318 if entry.get('abi') == 'tri-mode-v2' else 311
        require([(o.domain,o.version) for o in model.opset_import] == [('',17)], 'Expected ONNX17')
        require([(v.name,tensor_shape(v)) for v in (*model.graph.input,*model.graph.output)] ==
                [('obs',(TensorProto.FLOAT,[1,width])),('q_des',(TensorProto.FLOAT,[1,6]))], 'Unexpected policy ABI')
        c = Compiler(width)
        code, weights = c.compile(model)
        code_name, weight_name = f'graph-{digest(code)[:20]}.mjs', f'weights-{digest(weights)[:20]}.bin'
        (compiled/code_name).write_bytes(code)
        (compiled/weight_name).write_bytes(weights)
        meta = dict(format=1, graph=f'models/js/{code_name}', weights=f'models/js/{weight_name}',
                    sha256=digest(weights), graph_sha256=digest(code), onnx_sha256=entry['sha256'],
                    input=width, output=6, scratch=c.scratch, floats=c.weight_size)
        # JSON describes the static graph ABI/layout; topology is the referenced AOT module.
        descriptor = (json.dumps(meta,separators=(',',':'))+'\n').encode()
        descriptor_hash = digest(descriptor)
        filename = f'policy-{descriptor_hash[:20]}.graph.json'
        (compiled/filename).write_bytes(descriptor)
        entry['js'] = dict(path=f'models/js/{filename}',sha256=descriptor_hash)
        print(name,c.scratch,len(code),len(weights))
    manifest.write_text(json.dumps(bundle,indent=2)+'\n')


if __name__ == '__main__':
    main()
