/** Source-faithful calendar feature derivation, entirely in memory.
 * Contract: recorded adult age, source-style booking/appointment calendar dates,
 * and appointment hospital/clinic neighbourhood. No patient identifiers.
 */
const INPUT_KEYS = ['age','appointment_date','booking_date','neighbourhood'];
const DAY_MS = 86400000;
function parseCalendarDate(value) {
  if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(value) || value.slice(0,4)==='0000') {
    throw new Error('Dates must be valid YYYY-MM-DD calendar dates');
  }
  const milliseconds=Date.parse(value+'T00:00:00.000Z');
  if (!Number.isFinite(milliseconds) || new Date(milliseconds).toISOString().slice(0,10)!==value) {
    throw new Error('Dates must be valid YYYY-MM-DD calendar dates');
  }
  return new Date(milliseconds);
}
export function toModelFeatures(model,input) {
  if (!input || typeof input !== 'object' || Array.isArray(input) ||
      Object.keys(input).sort().join('|') !== INPUT_KEYS.join('|')) {
    throw new Error('Expected exactly age, appointment_date, booking_date, neighbourhood');
  }
  if (!Number.isInteger(input.age) || input.age < 18 || input.age > 110) {
    throw new Error('This source-specific adult model requires recorded age 18–110 in whole years');
  }
  if (input.neighbourhood !== null && typeof input.neighbourhood !== 'string') {
    throw new Error('Neighbourhood must be an appointment-location category or blank');
  }
  if (typeof input.neighbourhood === 'string' && input.neighbourhood.length > 80) {
    throw new Error('Appointment-location category is too long');
  }
  const appointment=parseCalendarDate(input.appointment_date);
  const booking=parseCalendarDate(input.booking_date);
  const lead=(appointment.getTime()-booking.getTime())/DAY_MS;
  if (!Number.isInteger(lead) || lead < 3) {
    throw new Error('The frozen cohort requires at least three calendar days from booking to appointment');
  }
  const features={
    Age:input.age,
    lead_days:lead,
    log1p_lead_days:Math.log1p(lead),
    appointment_weekday:String((appointment.getUTCDay()+6)%7),
    booking_weekday:String((booking.getUTCDay()+6)%7),
    appointment_month:String(appointment.getUTCMonth()+1),
    Neighbourhood:input.neighbourhood,
  };
  const warnings=[];
  for (const name of model.preprocessing.categorical) {
    const value=features[name];
    if (value===null || value==='') warnings.push(name+': missing category encoded explicitly; missing inputs were absent in the frozen final cohort');
    else if (!model.preprocessing.vocab[name].includes(value)) warnings.push(name+': unseen category encoded as unknown');
  }
  if (input.appointment_date<'2016-04-29' || input.appointment_date>'2016-05-25') {
    warnings.push('Appointment date is outside the frozen fitting period, 2016-04-29 through 2016-05-25');
  }
  warnings.push('Historical Vitória cohort only; applicability to new places, periods, or care settings is unvalidated');
  return {features,warnings};
}
