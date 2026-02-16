#![allow(incomplete_features)]
use std::collections::HashMap;

use itertools::Itertools;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use crate::{
    blocks::TitleLevel,
    entities::{Element, ElementID, PageID},
};

/// Minimum gap between headings to consider them in separate buckets
const TITLE_MERGE_THRESHOLD: f32 = 0.7;

pub fn title_levels_kmeans(
    titles: &[&Element],
    title_buckets: usize,
) -> HashMap<(PageID, ElementID), TitleLevel> {
    let mut title_level = HashMap::new();

    let samples: Vec<f32> = titles.iter().map(|e| e.bbox.height()).collect();
    let sample_len = samples.len();

    // TODO: Check this heuristic
    if sample_len <= title_buckets {
        return title_level;
    }

    let (centroids, assignments) = run_kmeans_1d(&samples, title_buckets, 100);

    let centroids_sorted: Vec<_> = centroids
        .iter()
        .enumerate()
        .map(|(c_idx, c)| (c_idx, *c))
        .sorted_by(|(_, c1), (_, c2)| c2.total_cmp(c1))
        .collect();

    let mut centroid_mapping = vec![-1i8; centroids_sorted.len()];

    // Map sorted centroids to levels, merging close ones
    let mut prev_centroid = (1, centroids_sorted[0].1);
    for (c_idx, c_val) in centroids_sorted.iter() {
        if *c_val < prev_centroid.1 * TITLE_MERGE_THRESHOLD {
            prev_centroid.0 += 1;
        }
        centroid_mapping[*c_idx] = prev_centroid.0;
        prev_centroid.1 = *c_val;
    }

    for (el, assignment) in titles.iter().zip(assignments.iter()) {
        assert!(centroid_mapping[*assignment] >= 0);
        title_level.insert((el.page_id, el.id), centroid_mapping[*assignment] as u8);
    }

    title_level
}

/// Simple 1D K-Means implementation with K-Means++ initialization
fn run_kmeans_1d(samples: &[f32], k: usize, max_iters: usize) -> (Vec<f32>, Vec<usize>) {
    let n = samples.len();
    if n == 0 || k == 0 {
        return (vec![], vec![]);
    }
    if k >= n {
        // If k >= n, just use samples as centroids
        let assignments: Vec<usize> = (0..n).collect();
        return (samples.to_vec(), assignments);
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut centroids = Vec::with_capacity(k);

    // K-Means++ Initialization
    // 1. Choose one center uniformly at random from among the data points.
    if let Some(&first) = samples.choose(&mut rng) {
        centroids.push(first);
    } else {
        return (vec![0.0; k], vec![0; n]); // Should not happen given check above
    }

    // 2. For each data point x, compute D(x), the distance between x and the nearest center that has already been chosen.
    // 3. Choose one new data point at random as a new center, using a weighted probability distribution where a point x is chosen with probability proportional to D(x)^2.
    // 4. Repeat Steps 2 and 3 until k centers have been chosen.
    for _ in 1..k {
        let mut dists_sq = Vec::with_capacity(n);
        let mut sum_dist_sq = 0.0;

        for &x in samples {
            let min_dist_sq = centroids
                .iter()
                .map(|&c| (x - c).powi(2))
                .fold(f32::INFINITY, |a, b| a.min(b));
            dists_sq.push(min_dist_sq);
            sum_dist_sq += min_dist_sq;
        }

        if sum_dist_sq <= f32::EPSILON {
            // All points are on top of existing centroids, pick random unpicked point or just random
            if let Some(&next) = samples.choose(&mut rng) {
                centroids.push(next);
            }
        } else {
            let target = rng.gen::<f32>() * sum_dist_sq;
            let mut current_sum = 0.0;
            let mut chosen = samples[n - 1]; // Fallback
            for (i, &d) in dists_sq.iter().enumerate() {
                current_sum += d;
                if current_sum >= target {
                    chosen = samples[i];
                    break;
                }
            }
            centroids.push(chosen);
        }
    }

    let mut assignments = vec![0; n];
    let mut old_assignments = vec![0; n];

    for _iter in 0..max_iters {
        // Assignment step
        let mut changed = false;
        for (i, &x) in samples.iter().enumerate() {
            let mut min_dist_sq = f32::INFINITY;
            let mut best_cluster = 0;
            for (j, &c) in centroids.iter().enumerate() {
                let d = (x - c).powi(2);
                if d < min_dist_sq {
                    min_dist_sq = d;
                    best_cluster = j;
                }
            }
            if assignments[i] != best_cluster {
                assignments[i] = best_cluster;
                changed = true;
            }
        }

        if !changed && _iter > 0 {
            break;
        }

        // Update step
        let mut new_centroids = vec![0.0; k];
        let mut counts = vec![0; k];

        for (i, &cluster) in assignments.iter().enumerate() {
            new_centroids[cluster] += samples[i];
            counts[cluster] += 1;
        }

        for j in 0..k {
            if counts[j] > 0 {
                centroids[j] = new_centroids[j] / counts[j] as f32;
            } else {
                // If a cluster becomes empty, reinitialize it to a random point (or keep old, or pick point furthest from any centroid)
                // For simplicity, just pick a random point to restart it
                if let Some(&p) = samples.choose(&mut rng) {
                    centroids[j] = p;
                }
            }
        }

        old_assignments.copy_from_slice(&assignments);
    }

    (centroids, assignments)
}
