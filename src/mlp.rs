use nalgebra::DMatrix;

use crate::{
    functions,
    utils::{hadamard, new_matrix},
};

pub struct Mlp {
    pub input_size: usize,
    pub layers: Vec<usize>,
    pub weights: Vec<DMatrix<f64>>,
    activation_fn: fn(f64) -> f64,
    activation_derivative_fn: fn(f64) -> f64,
    pub loss_fn: fn(&DMatrix<f64>, &DMatrix<f64>) -> f64,
    loss_derivative_fn: fn(&DMatrix<f64>, &DMatrix<f64>) -> DMatrix<f64>,
    pub hyper_parameters: HyperParameters,
}

pub struct ForwardPass {
    pub inputs: DMatrix<f64>,
    pub activation_values: Vec<DMatrix<f64>>,
    pub z_values: Vec<DMatrix<f64>>,
}

pub struct HyperParameters {
    pub gradient_clipping: f64,
    pub learning_rate: f64,
}

impl ForwardPass {
    /// Get the raw logits from the final layer
    ///
    /// Although we calculate activation values in the final layer,
    /// we actually want the raw logits. This is because if we do something
    /// like leaky ReLU, then our probability distribution will get messed up
    /// and we'll be unable to send a strong negative signal. Therefore, we ignore
    /// the activation values of the last layer, effectively removing it from the neural
    /// network. You can probably optimize this code to have the last layer not calculate the
    /// activations, but it's honestly negligible performance-wise
    pub fn result(&self) -> &DMatrix<f64> {
        self.z_values.last().unwrap()
    }
    pub fn loss(&self, loss_fn: fn(&DMatrix<f64>, &DMatrix<f64>) -> f64, y: &DMatrix<f64>) -> f64 {
        (loss_fn)(self.result(), y)
    }
}

impl Mlp {
    /// Create a new MLP
    pub fn new(input_size: usize, layers: Vec<usize>, hyper_parameters: HyperParameters) -> Self {
        assert!(layers.len() > 1);
        // First Layer (need to size it for the input size)
        let mut weights = vec![new_matrix(layers[0], input_size, 0)];
        // Every other layer
        for (i, size) in layers.iter().enumerate() {
            // Skip the last index
            if i < layers.len() - 1 {
                weights.push(new_matrix(layers[i + 1], *size, i as u64));
            }
        }
        Mlp {
            input_size,
            layers,
            weights,
            hyper_parameters,
            // These were all chosen because I googled around a lot
            // and all the pytorch tutorials seem to choose these for MNIST classification
            activation_fn: functions::leaky_relu,
            activation_derivative_fn: functions::leaky_relu_derivative,
            loss_fn: functions::cross_entropy_loss_softmax,
            loss_derivative_fn: functions::cross_entropy_loss_softmax_derivative,
        }
    }

    /// Run a forward pass through the MLP
    pub fn forward(&self, inputs: &DMatrix<f64>) -> ForwardPass {
        // Do a forward pass
        let activate = self.activation_fn;
        let mut prev_layer = inputs.clone();

        let mut activation_values: Vec<DMatrix<f64>> = Vec::new();
        let mut z_values: Vec<DMatrix<f64>> = Vec::new();

        for i in 0..self.layers.len() {
            // Neural networks are just matrix multiplication at the end of the day,
            // https://www.gilesthomas.com/2025/02/basic-neural-network-matrix-maths-part-1
            // To really hammer this home, you should draw some NN's with different layer
            // sizes on paper, and then look at how they multiply together over weights.
            // It really cements how the intuitive NN representation maps to matrices
            let z = &self.weights[i] * &prev_layer;
            let a = z.map(&activate);
            z_values.push(z);
            prev_layer = a.clone();
            activation_values.push(a);
        }
        ForwardPass {
            inputs: inputs.clone(),
            activation_values,
            z_values,
        }
    }

    /// Run Backpropogation on the MLP, updates in place
    pub fn backward(&mut self, fwd: &ForwardPass, y: &DMatrix<f64>) {
        // First retrieve the result from the forward pass
        let result = fwd.result();
        // Loss Function
        // This ties all the final neurons down to a single
        // end neuron which gives one differentiable number that can be
        // used to calculate all the gradients in the weights
        // This is gradient of the loss function with respect to the raw output of the final layer (∂C/∂a)
        // NOTE: We use the raw output, not the activations, of the final layer (see comments in ForwardPass.result)
        let mut a_d = (self.loss_derivative_fn)(result, y);

        // Back propogation
        let nn_len = self.layers.len();
        for (i_forward, _curr_layer_size) in (self.layers).iter().rev().enumerate() {
            // Get the reverse index so we can iterate backwards
            let i = nn_len - (i_forward + 1);

            let (prev_a, _prev_layer_size) = match i {
                0 => (&fwd.inputs, self.input_size),
                _ => (&fwd.activation_values[i - 1], self.layers[i - 1]),
            };
            let z = &fwd.z_values[i];
            // derivative of activation function (∂a/∂z)
            let z_d = z.map(|x| (self.activation_derivative_fn)(x));

            // Use the hadamard product to apply the chain rule: ∂L/∂z = ∂L/∂a * ∂a/∂z
            let layer_d = hadamard(&a_d, &z_d);

            // Weight derivative ∂L/∂w
            // ∂L/∂w = ∂z/∂w * ∂L/∂a * ∂a/∂z
            // this is the machine learning magic.
            // If you look at how the weight matrix maps the previous layer to the first,
            // the relationship is for some weight Wxy where:
            // x is the index of the front layer
            // y is the index of the back layer
            // (remember that we index things backwards in the world of NN)
            // and we also know that the derivative of the weight function is just the value of
            // the activation function behind it (because you just multiply the weight value by the activation function output)
            // then via the chain rule you can get the gradient for the weight with respect to the loss function by
            // multiplying the output of the activation function from the previous neuron with the gradient of the front neuron.
            // Then, because we pass multiple samples through at once, we take the average gradient for the weight by summing all the
            // sample gradients together and then dividing by the number of samples (each sample results in another column of activation values)
            let w_d: DMatrix<f64> =
                // Notice that we use the front layers number of rows and the
                // previous layers number of rows to correctly model the weight matrix
                // If this doesn't make sense, go back to drawing NN's on paper until it sinks in
                DMatrix::from_fn(layer_d.nrows(), prev_a.nrows(), |row, col| {
                    // this is the ∂L/∂a * ∂a/∂z which we already calculated
                    let front_neuron_gradients = layer_d.row(row);
                    // ∂z/∂w
                    let back_neuron_activation_values = prev_a.row(col);
                    // If you send multiple samples through, then the adjustment to the weight is the average
                    // of all the gradients
                    let mut avg_weight_gradient = 0.0;
                    for (i, val) in front_neuron_gradients.iter().enumerate() {
                        avg_weight_gradient += val * back_neuron_activation_values[i];
                    }
                    avg_weight_gradient / (layer_d.ncols() as f64)
                });

            // NUM_STABLE
            // Gradient clipping was necessary here to prevent exploding gradients
            // it basically caps how aggressively we can adjust a weight
            // I don't have a link for this because it made sense to do on my own
            // and then later on I discovered there's an actual term for the practice.
            let w_d = w_d.map(|grad| {
                let clip_value = self.hyper_parameters.gradient_clipping;
                if grad > clip_value {
                    clip_value
                } else if grad < -clip_value {
                    -clip_value
                } else {
                    grad
                }
            });

            // Adjust the weights
            // Running this line is what makes the machine learn
            self.weights[i] = &self.weights[i] - (w_d * self.hyper_parameters.learning_rate);

            // Now we need to calculate ∂L/∂a^(prev) so we can repeat the steps above for the next layer back
            // By the chain rule (watch the 3b1b video to see this visually)
            // ∂L/∂a^(prev) = ∂L/∂a^(current) * ∂a^(current)/∂z^(current) * ∂z^(current)/∂a^(prev)
            // The difference between this and w_d is we're doing  ∂z^(current)/∂a^(prev) instead of  ∂z^(current)/∂w^(current)
            // we already have ∂L/∂a^(current) * ∂a^(current)/∂z^(current) in the variable layer_d
            // so we just need ∂z^(current)/∂a^(prev)
            // since we're taking the partial derivative of z with respect to a^(prev),
            // that means we just need to sum the weights that are attached to it
            a_d = DMatrix::from_fn(prev_a.nrows(), prev_a.ncols(), |row, col| {
                let mut activation_gradient = 0.0;
                // Every column is a vector of weights attached to this neuron
                // We need to summate the weights, multiplied by the layer derivative in front of it
                for front_neuron_row in 0..layer_d.nrows() {
                    let front_neuron_gradient = layer_d[(front_neuron_row, col)];
                    // The column vector in the weights matrix that is the one coming out
                    // of the previous neuron is the one whose column number matches that
                    // neurons row number
                    let weight = self.weights[i][(front_neuron_row, row)];
                    activation_gradient += weight * front_neuron_gradient;
                }
                activation_gradient
            });
        }
    }
}
