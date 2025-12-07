use crate::{
    mlp::{HyperParameters, Mlp},
    mnist::Dataset,
    utils::print_stats,
};
mod functions;
mod mlp;
mod mnist;
mod utils;

fn main() {
    let training_dataset = Dataset::from_files(
        "Training Data",
        "./data/train-images-idx3-ubyte",
        "./data/train-labels-idx1-ubyte",
        100,
        600,
    );

    let test_dataset = Dataset::from_files(
        "Test Data",
        "./data/t10k-images-idx3-ubyte",
        "./data/t10k-labels-idx1-ubyte",
        10_000,
        1,
    );

    let mut mlp = Mlp::new(
        28 * 28,
        vec![256, 64, 10],
        HyperParameters {
            gradient_clipping: 1.0,
            learning_rate: 0.03,
        },
    );
    // Print the initial model performance (should be ~10% for random guessing)
    print_stats(&mlp, &training_dataset);
    print_stats(&mlp, &test_dataset);

    let mut fwd = mlp.forward(&training_dataset.batches[0].input_data);
    // This is useful if you want to see a single sample of the output
    //println!("Guessing for: {}", dataset.batches[0].labels[0]);
    //println!("Starting Probabilities: {}", softmax(fwd.result()).column(0));
    let mut loss = fwd.loss(mlp.loss_fn, &training_dataset.batches[0].label_data);
    println!("Initial Loss {}", loss);
    println!("Training...");
    let num_epochs = 10;
    for epoch in 0..num_epochs {
        for batch in training_dataset.batches.iter() {
            fwd = mlp.forward(&batch.input_data);
            mlp.backward(&fwd, &batch.label_data);
        }
        fwd = mlp.forward(&training_dataset.batches[0].input_data);
        loss = fwd.loss(mlp.loss_fn, &training_dataset.batches[0].label_data);
        println!("(Epoch {}/{}) Loss: {}", epoch + 1, num_epochs, loss);
    }
    // This is useful if you want to see the same single sample of the output after training
    //fwd = mlp.forward(&dataset.batches[0].input_data);
    //println!("Result: {}", softmax(fwd.result()).column(0));

    // Print the final model performance
    print_stats(&mlp, &training_dataset);
    print_stats(&mlp, &test_dataset);
}
