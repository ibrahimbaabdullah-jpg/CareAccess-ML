import {validateModel, predict} from './predictor.mjs';
import {toModelFeatures} from './input-adapter.mjs';

const MODEL_SHA256 = '4e12a56458e354595638b2882f4e00d70086570f2b02727e6430a01b69e84dcc';
const MODEL_BYTES = 99222;
const $ = id => document.getElementById(id);
const form = $('research-form');
const inputIds = ['age', 'booking-date', 'appointment-date', 'neighbourhood'];
const fields = inputIds.map($);
const calculate = $('calculate');
let model = null;
let activeView = 'research';

function announce(message) { $('app-status').textContent = message; }
function clearResult() {
  announce('');
  $('result').hidden = true;
  $('empty-result').hidden = false;
  for (const id of ['score-display', 'exact-score', 'lead-value', 'warnings', 'features']) {
    $(id).replaceChildren();
  }
  document.querySelectorAll('.feature-details').forEach(el => { el.open = false; });
}
function clearError() {
  $('form-error').textContent = '';
  $('form-error').hidden = true;
  fields.forEach(field => field.removeAttribute('aria-invalid'));
}
function clearAll(notify = false) {
  form.reset();
  fields.forEach(field => { field.value = ''; field.disabled = false; });
  $('missing-location').checked = false;
  $('neighbourhood').required = true;
  clearResult();
  clearError();
  if (notify) announce('All inputs and results cleared.');
  else announce('');
}
function showError(message, id = null) {
  clearResult();
  $('form-error').textContent = message;
  $('form-error').hidden = false;
  if (id) {
    $(id).setAttribute('aria-invalid', 'true');
    $(id).focus();
  }
}
function readInput() {
  const age = $('age').value;
  if (!/^(?:[1-9][0-9]|1[01][0-9])$/.test(age) || Number(age) < 18 || Number(age) > 110) {
    showError('Enter recorded age as a whole number from 18 to 110.', 'age');
    return null;
  }
  for (const id of ['booking-date', 'appointment-date']) {
    if (!/^\d{4}-\d{2}-\d{2}$/.test($(id).value)) {
      showError('Enter both complete, valid calendar dates.', id);
      return null;
    }
  }
  const missing = $('missing-location').checked;
  const place = $('neighbourhood').value;
  if (!missing && (!place || place.length > 80 || place.trim() !== place || /[\p{Cc}\p{Cf}]/u.test(place))) {
    showError('Enter the exact hospital/clinic neighborhood category, without leading or trailing spaces or control characters, or mark it as not recorded.', 'neighbourhood');
    return null;
  }
  return {age:Number(age), booking_date:$('booking-date').value,
    appointment_date:$('appointment-date').value, neighbourhood:missing ? null : place};
}
const featureNames = {Age:'Recorded age', lead_days:'Booking lead days', log1p_lead_days:'log(1 + lead days)', appointment_weekday:'Appointment weekday (Monday = 0)', booking_weekday:'Booking weekday (Monday = 0)', appointment_month:'Appointment month', Neighbourhood:'Hospital / clinic neighborhood'};
function renderResult(adapted, prediction) {
  if (!Number.isFinite(prediction.score) || prediction.score < 0 || prediction.score > 1) {
    throw new Error('The model did not produce a finite score. No result is available.');
  }
  $('score-display').textContent = (prediction.score * 100).toFixed(2) + '%';
  $('exact-score').textContent = String(prediction.score);
  $('lead-value').textContent = String(adapted.features.lead_days) + ' calendar days';
  const warnings = ['Raw scores overpredicted recorded no-shows in the final historical cohort. They have not been recalibrated.'];
  for (const warning of adapted.warnings) {
    warnings.push(warning.replace('Neighbourhood:', 'Hospital/clinic neighborhood:').replace('appointment_month:', 'Appointment month:').replace('appointment_weekday:', 'Appointment weekday:').replace('booking_weekday:', 'Booking weekday:'));
  }
  for (const text of warnings) {
    const li = document.createElement('li'); li.textContent = text; $('warnings').append(li);
  }
  for (const [key, value] of Object.entries(adapted.features)) {
    const dt = document.createElement('dt'); dt.textContent = featureNames[key];
    const dd = document.createElement('dd'); dd.textContent = value === null ? 'Explicitly missing' : String(value);
    $('features').append(dt, dd);
  }
  $('empty-result').hidden = true;
  $('result').hidden = false;
  announce('Research output ready: ' + $('score-display').textContent + '. This is an uncalibrated score, not an individual risk estimate. Read all interpretation warnings.');
}
form.addEventListener('submit', event => {
  event.preventDefault();
  clearResult(); clearError();
  if (!model) { showError('The frozen model is unavailable. Reload the page to retry.'); return; }
  const input = readInput();
  if (!input) return;
  try {
    const adapted = toModelFeatures(model, input);
    renderResult(adapted, predict(model, adapted.features));
  } catch (error) {
    showError(error instanceof Error ? error.message : 'The input could not be processed.');
  }
});
form.addEventListener('input', () => { clearResult(); clearError(); announce('Previous output cleared. Recalculate after editing.'); });
form.addEventListener('change', () => { clearResult(); clearError(); });
$('missing-location').addEventListener('change', () => {
  const missing = $('missing-location').checked;
  $('neighbourhood').value = '';
  $('neighbourhood').disabled = missing;
  $('neighbourhood').required = !missing;
});
$('clear').addEventListener('click', () => { clearAll(true); $('age').focus(); });

function selectView() {
  const requested = location.hash.slice(1);
  const next = ['research','evidence','privacy'].includes(requested) ? requested : 'research';
  if (next !== activeView) clearAll();
  activeView = next;
  document.querySelectorAll('[data-panel]').forEach(panel => { panel.hidden = panel.dataset.panel !== next; });
  document.querySelectorAll('[data-view]').forEach(link => {
    if (link.dataset.view === next) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  });
}
window.addEventListener('hashchange', () => { selectView(); $('main').focus({preventScroll:true}); window.scrollTo(0,0); });
window.addEventListener('pagehide', () => clearAll());
window.addEventListener('pageshow', () => { clearAll(); selectView(); });
clearAll();
selectView();

async function loadModel() {
  try {
    const response = await fetch(new URL('./frozen_xgb_d3.inference.json', import.meta.url), {
      method:'GET', credentials:'omit', cache:'no-store', referrerPolicy:'no-referrer', redirect:'error'
    });
    if (!response.ok) throw new Error('Model response failed');
    const bytes = await response.arrayBuffer();
    if (bytes.byteLength !== MODEL_BYTES) throw new Error('Model size differs from frozen asset');
    if (!globalThis.crypto?.subtle) throw new Error('A secure origin is required for model verification');
    const digest = await crypto.subtle.digest('SHA-256', bytes);
    const hex = [...new Uint8Array(digest)].map(value => value.toString(16).padStart(2, '0')).join('');
    if (hex !== MODEL_SHA256) throw new Error('Model integrity verification failed');
    const parsed = JSON.parse(new TextDecoder('utf-8', {fatal:true}).decode(bytes));
    validateModel(parsed);
    const fragment = document.createDocumentFragment();
    for (const value of parsed.preprocessing.vocab.Neighbourhood) {
      const option = document.createElement('option'); option.value = value; fragment.append(option);
    }
    $('neighbourhood-options').replaceChildren(fragment);
    model = parsed;
    $('model-status').textContent = 'Frozen model ready';
    calculate.textContent = 'Calculate research output';
    calculate.disabled = false;
  } catch {
    model = null; clearAll();
    $('model-status').textContent = 'Model unavailable';
    $('model-status').classList.add('failed');
    calculate.textContent = 'Model unavailable';
    calculate.disabled = true;
    showError('The frozen model could not be loaded and verified. Reload this page to retry. A secure connection or a localhost static server is required.');
  }
}
loadModel();
