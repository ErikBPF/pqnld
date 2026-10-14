// Deterministic label binding for the lettered readout.
//
// Options are labelled in option order today, which confounds the label bound
// to an option with that option's position: a score change under reordering
// cannot be attributed to either. A seeded cyclic shift decouples them. Over `k`
// consecutive seeds each label occupies each position exactly once, which is the
// counterbalancing the label-bias measurement needs; the shift is pure, so the
// same seed always renders the same binding and seed `0` (or any multiple of
// `k`) is the historical in-order binding.

/// Return `labels` cyclically shifted left by `seed % labels.len()`.
///
/// The returned vector is the same length and multiset as the input, so the
/// caller's option-order indexing stays valid: `out[i]` is the label rendered
/// for option `i`.
pub fn permuted(labels: Vec<String>, seed: u64) -> Vec<String> {
    let mut out = labels;
    if out.len() > 1 {
        let offset = (seed % out.len() as u64) as usize;
        out.rotate_left(offset);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::permuted;

    fn labels() -> Vec<String> {
        ["a", "b", "c", "d"].iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn same_seed_renders_the_same_binding() {
        assert_eq!(permuted(labels(), 7), permuted(labels(), 7));
    }

    #[test]
    fn zero_seed_is_the_in_order_binding() {
        assert_eq!(permuted(labels(), 0), labels());
        assert_eq!(permuted(labels(), 4), labels());
    }

    #[test]
    fn each_label_occupies_each_position_once_over_k_seeds() {
        let k = 4usize;
        let mut seen = vec![vec![false; k]; k];
        for seed in 0..k as u64 {
            for (position, label) in permuted(labels(), seed).iter().enumerate() {
                let label_index = labels().iter().position(|l| l == label).unwrap();
                assert!(
                    !seen[position][label_index],
                    "label {label} twice at position {position}"
                );
                seen[position][label_index] = true;
            }
        }
        assert!(seen.iter().all(|row| row.iter().all(|v| *v)));
    }

    #[test]
    fn the_binding_is_a_permutation_of_the_supplied_labels() {
        let mut sorted = permuted(labels(), 3);
        sorted.sort();
        assert_eq!(sorted, labels());
    }
}
