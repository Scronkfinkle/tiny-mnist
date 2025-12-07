use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
};

use nalgebra::DMatrix;

pub struct Batch {
    pub input_data: DMatrix<f64>,
    pub label_data: DMatrix<f64>,
    pub labels: Vec<i32>,
}

pub struct Dataset {
    pub name: String,
    pub batches: Vec<Batch>,
}

impl Dataset {
    // Writes out one of the MNIST samples to a PPM so you can see the handwritten digit
    //
    // fn write_img(img_data: &Vec<u8>, filename: &str) -> std::io::Result<()> {
    //     let mut output = File::create(filename).unwrap();
    //     output.write_all(b"P6\n28 28\n255\n")?;

    //     for val in img_data {
    //         output.write_all(&[*val])?;
    //         output.write_all(&[*val])?;
    //         output.write_all(&[*val])?;
    //     }
    //     Ok(())
    // }

    /// Turn a label for a digit (0-9) into a 1-hot encoded vector
    fn labels_to_matrix(labels: &Vec<i32>) -> DMatrix<f64> {
        DMatrix::from_fn(10, labels.len(), |row, col| {
            let label = labels[col];
            if (row as i32) == label {
                return 1.0;
            }
            0.0
        })
    }

    /// Load MNIST data from a file
    pub fn from_files(
        name: &str,
        data_path: &str,
        label_path: &str,
        batch_size: usize,
        num_batches: usize,
    ) -> Self {
        // Loads the data from the MNIST dataset
        let mut batches: Vec<Batch> = Vec::new();
        let img_size = 28 * 28;
        let mut f_img = File::open(data_path).unwrap();
        let mut f_label = File::open(label_path).unwrap();

        f_img.seek(SeekFrom::Start(16)).unwrap();
        f_label.seek(SeekFrom::Start(8)).unwrap();

        for _j in 0..num_batches {
            let mut img_data: Vec<f64> = Vec::new();
            let mut labels: Vec<i32> = Vec::new();

            for _i in 0..batch_size {
                let mut img_buf = vec![0; img_size];
                let mut label_buf = vec![0; 1];

                f_img.read_exact(&mut img_buf).unwrap();
                f_label.read_exact(&mut label_buf).unwrap();

                //let filename = format!("renders/sample_{}.ppm");
                //println!("sample_{}: {}", _i, label_buf[0]);
                for val in img_buf.iter() {
                    let val64 = *val as f64;
                    img_data.push(val64);
                }
                let label64 = label_buf[0] as i32;
                labels.push(label64);
                // Uncomment to write the data to an image file so you can see the
                // handwritten number
                //Self::write_img(&img_buf, &filename).unwrap();
            }

            let input_data = DMatrix::from_fn(img_size, batch_size, |row, col| {
                let seek = col * img_size;
                img_data[seek + row] / 255.0
            });
            let label_data = Self::labels_to_matrix(&labels);
            batches.push(Batch {
                input_data,
                label_data,
                labels,
            });
        }
        Dataset {
            batches,
            name: String::from(name),
        }
    }
}
