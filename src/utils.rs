use nalgebra::DMatrix;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Normal};

use crate::{mlp::Mlp, mnist::Dataset};

/// Generate a matrix of arbitrary size with initialization weights
pub fn new_matrix(nrows: usize, ncols: usize, seed: u64) -> DMatrix<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    DMatrix::from_fn(nrows, ncols, |_row, _col| {
        // NUM_STABLE
        // We do this instead of just rng::random because if we don't
        // then the weight values get way too big and turn into NaN's everywhere
        // There's a couple initialization functions out there. When I googled the first that was
        // suggested was Xavier Initialization which is the "common rule of thumb" of initializing
        // I stumbled upon this other method though, call He initialization, which is apparently better for
        // neural networks that reLU like activations
        // https://medium.com/@sanjay_dutta/understanding-glorot-and-he-initialization-a-guide-for-college-students-00f3dfae0393
        let std_dev = (2.0 / nrows as f64).sqrt() * 0.1; // NUM_STABLE multiply by 0.1 to make the weights a bit smaller, seems to help
        let normal = Normal::new(0.0, std_dev).unwrap();
        normal.sample(&mut rng)
    })
}

/// Compute the Hadamard product
///
/// This is a fancy word for element-wise multiplication between
/// two matrices. It shows up if you google the real backpropogation formulas.
/// While I never actually figured those formulas out, the ability to do
/// element-wise multiplication becomes extremely useful.
/// I learned about it here:
/// http://neuralnetworksanddeeplearning.com/chap2.html
/// but didn't really understand the full article
pub fn hadamard(a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    assert_eq!(a.nrows(), b.nrows());
    assert_eq!(a.ncols(), b.ncols());
    let nrows = a.nrows();
    let ncols = a.ncols();

    DMatrix::from_fn(nrows, ncols, |row, col| a[(row, col)] * b[(row, col)])
}

/// Nice debugging method for evaluating a dataset with an MLP
pub fn print_stats(mlp: &Mlp, dataset: &Dataset) {
    let mut correct: i32 = 0;
    let mut incorrect: i32 = 0;
    for batch in dataset.batches.iter() {
        let fwd = mlp.forward(&batch.input_data);
        for (i, column) in fwd.result().column_iter().enumerate() {
            let mut guess = 0;
            let mut max = 0.0;
            for (row_num, row_val) in column.iter().enumerate() {
                if *row_val > max {
                    max = *row_val;
                    guess = row_num;
                }
            }
            let answer = batch.labels[i];
            if (guess as i32) == answer {
                correct += 1;
            } else {
                incorrect += 1;
            }
        }
    }
    println!("# Model Performance ({})", dataset.name);
    println!(
        "# Correct: {}  Incorrect: {} Accuracy: {:.2}%",
        correct,
        incorrect,
        (correct as f32 / (correct + incorrect) as f32) * 100.0
    );
}
