use nalgebra::DMatrix;

use crate::utils::hadamard;

pub fn softmax(ak: &DMatrix<f64>) -> DMatrix<f64> {
    // Softmax lets you take an arbitray set of values and cast them
    // into a probability distribution. It's the go-to for neural nets
    // that do classification tasks.
    // https://www.geeksforgeeks.org/deep-learning/the-role-of-softmax-in-neural-networks-detailed-explanation-and-applications/
    let sums = ak.map(|e| e.exp()).row_sum();
    DMatrix::from_fn(ak.nrows(), ak.ncols(), |row, col| {
        ak[(row, col)].exp() / sums[col]
    })
}

// pub fn softmax_log_sum_exp(ak: &DMatrix<f64>) -> DMatrix<f64> {
//     //  NUM_STABLE
//     // https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
//     let mut lse_list = Vec::new();
//     for col in ak.column_iter() {
//         let max = col.max();
//         let logsumexp = col.map(|val| {
//             (val-max).exp()
//         }).sum().ln()+max;
//         lse_list.push(logsumexp);
//     }
//     DMatrix::from_fn(ak.nrows(), ak.ncols(), |row, col| {
//         (ak[(row,col)]-lse_list[col]).exp()
//     })
// }

pub fn cross_entropy_loss_softmax(ak: &DMatrix<f64>, y: &DMatrix<f64>) -> f64 {
    // If you read any pytorch tutorial for MNIST you'll likely see them
    // use cross-entropy as the loss function. That's because it's very good
    // for classification problems.
    // Here's a good video that explains cross entropy:
    // https://www.youtube.com/watch?v=KHVR587oW8I
    // The formula is sum(y_true * log(1/y_pred))
    // = sum(y_true * log(y_pred^-1))
    // = sum(y_true*(-1) * log(y_pred))
    // = -sum(y_true * log(y_pred))
    let mut result = softmax(ak);
    result = result.map(|entry| {
        // NUM_STABLE
        // Switched to ln over log10 to avoid hitting NaN's
        // I also add 1e-12 to prevent evaluation ln(0)
        // I arbtrarily chose that 1e-12
        (entry + 1e-12).ln()
    });
    result = hadamard(&result, y);
    // Return the average loss per sample (if more than one is used)
    -result.sum() / (y.ncols() as f64)
}

pub fn cross_entropy_loss_softmax_derivative(a_l: &DMatrix<f64>, y: &DMatrix<f64>) -> DMatrix<f64> {
    // In what I can only describe as an incredible fact of math and reality,
    // applying the chain rule for d(softmax)*d(entropy_loss)
    // is equivalent to simply doing y_pred - y_true
    // This saves you the pain of trying to differentiate the softmax and
    // cross-entropy functions separately.
    // https://www.geeksforgeeks.org/machine-learning/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss/
    a_l - y
}

pub fn leaky_relu(x: f64) -> f64 {
    // NUM_STABLE
    // Regular relU gives us dead neurons and
    // we get stuck. Leaky reLU is like reLU except
    // we can still return a very tiny negative signal
    // https://www.geeksforgeeks.org/machine-learning/Leaky-Relu-Activation-Function-in-Deep-Learning/
    if x < 0.0 {
        return x * 0.01;
    }
    x
}

pub fn leaky_relu_derivative(x: f64) -> f64 {
    // reLU in general is great because it's extremely easy to differentiate
    if x < 0.0 {
        return 0.01;
    }
    1.0
}

// Some Extra functions you may want to try
//
// pub fn tanh(x: f64) -> f64 {
//     let e: f64 = 2.71828;
//     (e.powf(2.0 * x) - 1.0) / (e.powf(2.0 * x) + 1.0)
// }

// pub fn tanh_derivative(x: f64) -> f64 {
//     1.0 - (tanh(x).powi(2))
// }

// pub fn reLU(x: f64) -> f64 {
//     if x < 0.0 {
//         return 0.0;
//     }
//     return x;
// }

// pub fn reLU_derivative(x: f64) -> f64 {
//     if x < 0.0 {
//         return 0.0;
//     }
//     return 1.0;
// }

// pub fn mse(aL: &DMatrix<f64>, y: &DMatrix<f64>) -> f64 {
//     let diff = aL - y;
//     let diff_squared = diff.map(|x| x.powi(2));
//     let mse = diff_squared.sum();

//     return mse;
// }

// pub fn mse_derivative(aL: &DMatrix<f64>, y: &DMatrix<f64>) -> DMatrix<f64> {
//     // dC/da^L_i = 2(a_i - y)
//     //let grad = (aL-y)*2.0;
//     //let res = DMatrix::from_columns(&[grad.column_sum()]) / (aL.ncols() as f64);
//     //return res;
//     (aL - y) * 2.0
// }
