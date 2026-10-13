#!/usr/bin/env python3
"""Dependency-free frozen-asset checks. No source records or model fitting."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import math
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SITE_PATHS = {
    '.nojekyll', 'LICENSE-CODE.txt', 'MODEL-ASSET-LICENSE.txt',
    'assets/MODEL_ASSET_NOTICE.txt', 'assets/app.mjs', 'assets/favicon.svg',
    'assets/frozen_xgb_d3.inference.json', 'assets/input-adapter.mjs',
    'assets/predictor.mjs', 'assets/styles.css', 'index.html',
}
VERIFICATION_PATHS = {
    'site-manifest.json', 'scripts/verify_static.py', 'VERIFICATION.md',
    '.github/workflows/static-check.yml',
}
MODEL_SHA256 = '4e12a56458e354595638b2882f4e00d70086570f2b02727e6430a01b69e84dcc'
SHA_PATTERN = re.compile(r'[0-9a-f]{64}')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key')
        result[key] = value
    return result


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'),
                      object_pairs_hook=no_duplicate_keys,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          ValueError('Nonfinite JSON constant')))


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def verify_files():
    manifest = read_json(ROOT / 'site-manifest.json')
    require(type(manifest) is list and len(manifest) == 11, 'Expected 11 site entries')
    seen = set()
    for entry in manifest:
        require(type(entry) is dict and set(entry) == {'path', 'bytes', 'sha256'},
                'Unexpected manifest fields')
        name = entry['path']
        require(type(name) is str and name in SITE_PATHS and name not in seen,
                'Unexpected or duplicate site path')
        require(not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts,
                'Unsafe path')
        seen.add(name)
        require(type(entry['bytes']) is int and entry['bytes'] >= 0, 'Invalid byte count')
        require(type(entry['sha256']) is str and SHA_PATTERN.fullmatch(entry['sha256']),
                'Invalid file checksum')
        path = ROOT / name
        require(path.is_file() and not path.is_symlink(), 'Missing or linked site file: ' + name)
        data = path.read_bytes()
        require(len(data) == entry['bytes'], 'Byte count changed: ' + name)
        require(hashlib.sha256(data).hexdigest() == entry['sha256'], 'Checksum changed: ' + name)
    require(seen == SITE_PATHS, 'Site manifest path mismatch')
    present = set()
    for path in ROOT.rglob('*'):
        relative = path.relative_to(ROOT)
        if relative.parts[0] == '.git':
            continue
        require(not path.is_symlink(), 'Symlinks are not permitted in the static release')
        if path.is_file():
            present.add(relative.as_posix())
    require(present == SITE_PATHS | VERIFICATION_PATHS, 'Unexpected or missing release files')
    model_bytes = (ROOT / 'assets/frozen_xgb_d3.inference.json').read_bytes()
    require(hashlib.sha256(model_bytes).hexdigest() == MODEL_SHA256, 'Frozen model changed')
    app = (ROOT / 'assets/app.mjs').read_text(encoding='utf-8')
    require("const MODEL_SHA256 = '" + MODEL_SHA256 + "';" in app, 'App model pin mismatch')
    require('const MODEL_BYTES = 99222;' in app, 'App model size pin mismatch')
    return len(manifest), sum(entry['bytes'] for entry in manifest)


def verify_model():
    model = read_json(ROOT / 'assets/frozen_xgb_d3.inference.json')
    require(type(model) is dict and set(model) == {
        'format', 'model_id', 'source_model_sha256', 'source_prediction_freeze_sha256',
        'objective', 'base_margin', 'preprocessing', 'trees'}, 'Unexpected model schema')
    require(model['format'] == 'careaccess-vitoria-inference-v1', 'Unknown format')
    require(model['model_id'] == 'frozen_xgb_d3', 'Unknown model identity')
    require(model['objective'] == 'binary:logistic', 'Unsupported objective')
    require(finite(model['base_margin']), 'Invalid base margin')
    for key in ['source_model_sha256', 'source_prediction_freeze_sha256']:
        require(type(model[key]) is str and SHA_PATTERN.fullmatch(model[key]), 'Invalid provenance hash')
    prep = model['preprocessing']
    require(type(prep) is dict and set(prep) == {
        'numeric', 'medians', 'categorical', 'vocab', 'n_features', 'scaled', 'history',
        'numeric_missing_indicators', 'categorical_missing_and_unknown'}, 'Unexpected preprocessing schema')
    require(prep['numeric'] == ['Age', 'lead_days', 'log1p_lead_days'], 'Numeric contract changed')
    require(type(prep['medians']) is list and len(prep['medians']) == 3
            and all(finite(x) for x in prep['medians']), 'Invalid medians')
    categorical = ['appointment_weekday', 'booking_weekday', 'appointment_month', 'Neighbourhood']
    require(prep['categorical'] == categorical, 'Categorical contract changed')
    require(prep['scaled'] is False and prep['history'] is False, 'Unsupported transforms')
    require(prep['numeric_missing_indicators'] is True
            and prep['categorical_missing_and_unknown'] is True, 'Encoding flags changed')
    require(type(prep['n_features']) is int and prep['n_features'] == 108, 'Feature width changed')
    require(type(prep['vocab']) is dict and set(prep['vocab']) == set(categorical), 'Vocabulary fields changed')
    for name, expected_size in zip(categorical, [6, 6, 2, 80]):
        vocab = prep['vocab'][name]
        require(type(vocab) is list and len(vocab) == expected_size, 'Vocabulary size changed')
        require(all(type(x) is str and 0 < len(x) <= 80 and not any(ord(c) < 32 for c in x)
                    for x in vocab), 'Invalid vocabulary value')
        require(len(set(vocab)) == len(vocab) and sorted(vocab) == vocab, 'Vocabulary ordering changed')
    require(2 * len(prep['numeric']) + sum(len(v) + 2 for v in prep['vocab'].values()) == 108,
            'Encoded width mismatch')
    trees = model['trees']
    require(type(trees) is list and len(trees) == 300, 'Tree count changed')
    total_nodes = 0
    expected_keys = {'left_children', 'right_children', 'split_indices', 'split_conditions', 'default_left'}
    for tree in trees:
        require(type(tree) is dict and set(tree) == expected_keys, 'Unexpected tree schema')
        require(all(type(v) is list for v in tree.values()), 'Tree arrays required')
        n = len(tree['left_children'])
        require(1 <= n <= 15 and all(len(v) == n for v in tree.values()), 'Invalid tree array widths')
        require(all(type(v) is int and 0 <= v < 108 for v in tree['split_indices']), 'Invalid feature index')
        require(all(finite(v) and abs(v) <= 3.4028234663852886e38 for v in tree['split_conditions']),
                'Invalid threshold or leaf weight')
        require(all(type(v) is int and v in (0, 1) for v in tree['default_left']), 'Invalid missing direction')
        incoming = [0] * n
        for node, (left, right) in enumerate(zip(tree['left_children'], tree['right_children'])):
            require(type(left) is int and type(right) is int, 'Noninteger child')
            if left == -1 or right == -1:
                require(left == right == -1, 'Incomplete leaf')
            else:
                require(0 <= left < n and 0 <= right < n and left != right
                        and left != node and right != node, 'Invalid child reference')
                incoming[left] += 1
                incoming[right] += 1
        require(incoming[0] == 0 and all(count == 1 for count in incoming[1:]), 'Invalid parent graph')
        visited = set()
        stack = [(0, 0)]
        while stack:
            node, depth = stack.pop()
            require(node not in visited and depth <= 3, 'Cycle or invalid tree depth')
            visited.add(node)
            if tree['left_children'][node] != -1:
                stack.extend([(tree['left_children'][node], depth + 1),
                              (tree['right_children'][node], depth + 1)])
        require(len(visited) == n, 'Unreachable tree nodes')
        total_nodes += n
    require(total_nodes == 3692, 'Total node count changed')
    return len(trees), total_nodes


def verify_javascript():
    for name in ['app.mjs', 'input-adapter.mjs', 'predictor.mjs']:
        subprocess.run(['node', '--check', str(ROOT / 'assets' / name)], check=True, cwd=ROOT)
    # This only loads/validates the real frozen model. It never scores a record.
    subprocess.run(['node', '--input-type=module', '-e',
        "import fs from 'node:fs'; import {validateModel} from './assets/predictor.mjs'; "
        "validateModel(JSON.parse(fs.readFileSync('./assets/frozen_xgb_d3.inference.json','utf8')));"],
        check=True, cwd=ROOT)


def main():
    files, size = verify_files()
    trees, nodes = verify_model()
    verify_javascript()
    print(f'PASS: {files} frozen site files, {size} bytes; {trees} trees, {nodes} nodes; 3 JavaScript syntax checks.')
    print('Static checks only. No input records, prediction parity, browser behavior, or clinical claims are tested.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print('FAIL: ' + str(error), file=sys.stderr)
        sys.exit(1)
