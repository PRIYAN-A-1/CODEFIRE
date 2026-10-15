export function joinLabels(labels, separator) {
  const cleaned = labels.map(label => label.trim());
  return cleaned.filter(label => label.length > 0).join(separator);
}
export function uniqueItems(items) {
  return Array.from(new Set(items));
}
