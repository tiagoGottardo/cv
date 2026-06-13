pub mod network {
    use ndarray::{Array1, Array2, s};
    use ndarray_rand::{
        RandomExt,
        rand::seq::SliceRandom,
        rand_distr::{Normal, NormalError},
    };

    pub struct Network {
        pub num_layers: usize,
        pub sizes: Array1<usize>,
        pub biases: Vec<Array2<f32>>,
        pub weights: Vec<Array2<f32>>,
    }

    impl Network {
        pub fn new(sizes: Vec<usize>) -> Result<Self, NormalError> {
            let sizes = Array1::from_vec(sizes);

            let num_layers = sizes.len();

            let distribution = Normal::new(0.0, 1.0)?;

            let biases: Vec<Array2<f32>> = sizes
                .slice(s![1..])
                .iter()
                .map(|y| Array2::random((*y as usize, 1 as usize), distribution))
                .collect();

            let weights: Vec<Array2<f32>> = sizes
                .windows(2)
                .into_iter()
                .map(|w| Array2::random((w[1] as usize, w[0] as usize), distribution))
                .collect();

            Ok(Self {
                num_layers,
                sizes,
                biases,
                weights,
            })
        }

        pub fn exec(
            &mut self,
            mut training_data: Vec<(Array2<f32>, usize)>,
            epochs: usize,
            mini_batch_size: usize,
            eta: f32,
            test_data: Option<Vec<(Array2<f32>, usize)>>,
        ) {
            for epoch in 0..epochs {
                training_data.shuffle(&mut ndarray_rand::rand::rng());
                let mini_batches: Vec<Vec<(Array2<f32>, usize)>> = training_data
                    .chunks(mini_batch_size)
                    .map(|c| c.to_vec())
                    .collect();

                for mini_batch in mini_batches.into_iter() {
                    self.update_mini_batch(mini_batch, eta);
                }

                match test_data {
                    Some(ref data) => {
                        println!(
                            "Epoch {}: {} / {}",
                            epoch,
                            self.evaluate(data.clone()),
                            data.len()
                        )
                    }
                    None => println!("Epoch {} complete", epoch),
                }
            }
        }

        pub fn update_mini_batch(&mut self, mini_batch: Vec<(Array2<f32>, usize)>, eta: f32) {
            let mut nabla_b: Vec<Array2<f32>> = self
                .biases
                .iter()
                .map(|b| Array2::zeros(b.raw_dim()))
                .collect();
            let mut nabla_w: Vec<Array2<f32>> = self
                .weights
                .iter()
                .map(|w| Array2::zeros(w.raw_dim()))
                .collect();

            let mini_batch_len = mini_batch.len();

            mini_batch
                .into_iter()
                .map(|(x, y)| {
                    let mut y_vector = Array2::zeros((10, 1));
                    y_vector[[y, 0]] = 1.0;
                    (x, y_vector)
                })
                .for_each(|(x, y)| {
                    let (delta_nabla_b, delta_nabla_w) = self.backprop(x, y);
                    nabla_b = nabla_b
                        .iter()
                        .zip(delta_nabla_b.iter())
                        .map(|(nb, dnb)| nb + dnb)
                        .collect();
                    nabla_w = nabla_w
                        .iter()
                        .zip(delta_nabla_w.iter())
                        .map(|(nw, dnw)| nw + dnw)
                        .collect();
                });

            self.biases = self
                .biases
                .iter()
                .zip(nabla_b.iter())
                .map(|(b, nb)| b - (eta / mini_batch_len as f32) * nb)
                .collect();
            self.weights = self
                .weights
                .iter()
                .zip(nabla_w.iter())
                .map(|(w, nw)| w - (eta / mini_batch_len as f32) * nw)
                .collect();
        }

        pub fn backprop(
            &self,
            x: Array2<f32>,
            y: Array2<f32>,
        ) -> (Vec<Array2<f32>>, Vec<Array2<f32>>) {
            let mut nabla_b: Vec<Array2<f32>> = self
                .biases
                .iter()
                .map(|b| Array2::zeros(b.raw_dim()))
                .collect();
            let mut nabla_w: Vec<Array2<f32>> = self
                .weights
                .iter()
                .map(|w| Array2::zeros(w.raw_dim()))
                .collect();

            let mut activation = x.clone();
            let mut activations = vec![x];
            let mut zs: Vec<Array2<f32>> = Vec::new();

            for (b, w) in self.biases.iter().zip(self.weights.iter()) {
                let z = w.dot(&activation) + b;
                zs.push(z.clone());
                activation = Network::sigmoid(z);
                activations.push(activation.clone());
            }

            let mut delta = Network::cost_derivative(activations.last().expect("1").clone(), y)
                * Network::sigmoid_prime(zs.last().expect("2").clone());

            nabla_b.pop();
            nabla_b.push(delta.clone());

            nabla_w.pop();
            nabla_w.push(delta.dot(&activations.iter().rev().nth(1).unwrap().t()));

            for l in 2..self.num_layers {
                let z = &zs[zs.len() - l]; // Pega o Z da camada atual (da direita para esquerda)
                let sp = Network::sigmoid_prime(z.clone());

                let w_index = self.weights.len() - l + 1;
                delta = self.weights[w_index].t().dot(&delta) * sp;

                let b_index = nabla_b.len() - l;
                let w_nabla_index = nabla_w.len() - l;

                nabla_b[b_index] = delta.clone();
                nabla_w[w_nabla_index] = delta.dot(&activations[activations.len() - l - 1].t());
            }

            (nabla_b, nabla_w)
        }

        pub fn feedfoward(&self, input: Array2<f32>) -> Array2<f32> {
            let mut a = input;
            self.biases
                .iter()
                .zip(self.weights.iter())
                .for_each(|(b, w)| a = Network::sigmoid(w.dot(&a) + b));

            a
        }

        pub fn evaluate(&self, test_data: Vec<(Array2<f32>, usize)>) -> usize {
            test_data
                .into_iter()
                .map(|(input, output)| {
                    let ((result, _), _) = self
                        .feedfoward(input)
                        .indexed_iter()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .unwrap();

                    if result == output { 1 } else { 0 }
                })
                .sum()
        }

        pub fn cost_derivative(output_activation: Array2<f32>, y: Array2<f32>) -> Array2<f32> {
            output_activation - y
        }

        pub fn sigmoid(z: Array2<f32>) -> Array2<f32> {
            z.map(|x| 1. / (1. + (-x).exp()))
        }

        pub fn sigmoid_prime(z: Array2<f32>) -> Array2<f32> {
            let s = Network::sigmoid(z);
            s.clone() * (1. - s)
        }
    }
}

pub mod dataset {
    use std::fs::File;
    use std::io::{self, Read};

    use ndarray::Array2;

    #[derive(Debug)]
    pub struct MnistData {
        pub training_data: Vec<(Array2<f32>, usize)>,
        pub test_data: Vec<(Array2<f32>, usize)>,
    }

    impl MnistData {
        pub fn load() -> io::Result<Self> {
            let training_data: Vec<(Array2<f32>, usize)> =
                read_images("assets/mnist/train-images.idx3-ubyte")?
                    .into_iter()
                    .zip(read_labels("assets/mnist/train-labels.idx1-ubyte")?.into_iter())
                    .map(|x| {
                        (
                            Array2::from_shape_vec((784, 1), x.0)
                                .expect("Error on conversion from Vec to ndarray"),
                            x.1 as usize,
                        )
                    })
                    .collect();

            let test_data: Vec<(Array2<f32>, usize)> =
                read_images("assets/mnist/t10k-images.idx3-ubyte")?
                    .into_iter()
                    .zip(read_labels("assets/mnist/t10k-labels.idx1-ubyte")?.into_iter())
                    .map(|x| {
                        (
                            Array2::from_shape_vec((784, 1), x.0)
                                .expect("Error on conversion from Vec to ndarray"),
                            x.1 as usize,
                        )
                    })
                    .collect();

            Ok(MnistData {
                training_data,
                test_data,
            })
        }
    }

    fn read_images(path: &str) -> io::Result<Vec<Vec<f32>>> {
        let mut file = File::open(path)?;
        let mut header = [0u8; 16];
        file.read_exact(&mut header)?;

        let num_images = u32::from_be_bytes([header[4], header[5], header[6], header[7]]) as usize;
        let mut all_images = Vec::with_capacity(num_images);
        let mut buffer = vec![0u8; 784];

        for _ in 0..num_images {
            file.read_exact(&mut buffer)?;
            let normalized: Vec<f32> = buffer.iter().map(|&p| p as f32 / 255.0).collect();
            all_images.push(normalized);
        }
        Ok(all_images)
    }

    fn read_labels(path: &str) -> io::Result<Vec<u8>> {
        let mut file = File::open(path)?;
        let mut header = [0u8; 8];
        file.read_exact(&mut header)?;

        let mut labels = Vec::new();
        file.read_to_end(&mut labels)?;
        Ok(labels)
    }
}
