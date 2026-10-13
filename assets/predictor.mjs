/** Auditable inference-only evaluator for one fixed dense float32 tree ensemble.
 * No fetch, storage, telemetry, data upload, or external dependencies.
 * This is a private prototype, not a clinical tool or publication approval.
 */
const f32 = Math.fround;

export function validateModel(model) {
  const p = model.preprocessing;
  if (model.format !== 'careaccess-vitoria-inference-v1' ||
      model.objective !== 'binary:logistic' || p.scaled || p.history ||
      p.n_features !== 108 || model.trees.length !== 300) {
    throw new Error('Unsupported frozen model contract');
  }
}

export function preprocess(model, row) {
  const p = model.preprocessing;
  const out = new Float32Array(p.n_features);
  for (let j = 0; j < p.numeric.length; j++) {
    const value = row[p.numeric[j]];
    // The API intentionally accepts JSON numbers/null, not pandas string coercion.
    if (value !== null && value !== undefined && typeof value !== 'number') {
      throw new Error('Numeric inputs must be numbers or null');
    }
    const missing = !Number.isFinite(value);
    out[j] = missing ? p.medians[j] : value;
    out[j + p.numeric.length] = missing ? 1 : 0;
  }
  let offset = p.numeric.length * 2;
  for (const name of p.categorical) {
    const value = row[name];
    if (value !== null && value !== undefined && typeof value !== 'string') {
      throw new Error('Categorical inputs must be strings or null');
    }
    const vocab = p.vocab[name];
    let index;
    if (value === null || value === undefined || value === '') index = vocab.length;
    else {
      index = vocab.indexOf(value);
      if (index < 0) index = vocab.length + 1;
    }
    out[offset + index] = 1;
    offset += vocab.length + 2;
  }
  if (offset !== p.n_features || !out.every(Number.isFinite)) {
    throw new Error('Nonfinite or incompatible feature vector');
  }
  return out;
}

export function predictMargin(model, features) {
  let margin = f32(model.base_margin);
  for (const tree of model.trees) {
    let node = 0;
    while (tree.left_children[node] !== -1) {
      const value = features[tree.split_indices[node]];
      const left = Number.isNaN(value) ? tree.default_left[node] : value < f32(tree.split_conditions[node]);
      node = left ? tree.left_children[node] : tree.right_children[node];
    }
    margin = f32(margin + f32(tree.split_conditions[node]));
  }
  return margin;
}

export function sigmoidFloat32(margin) {
  const exponential = f32(Math.exp(f32(-margin)));
  return f32(1 / f32(1 + exponential));
}

export function predict(model, row) {
  const features = preprocess(model,row);
  const margin = predictMargin(model,features);
  return {margin, score:sigmoidFloat32(margin)};
}
