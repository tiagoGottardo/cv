use anyhow::Result;

use computer_vision::cnn::{dataset::MnistData, network::Network};

fn main() -> Result<()> {
    let dataset = MnistData::load()?;

    let mut network = Network::new(vec![784, 30, 10])?;

    network.exec(dataset.training_data, 30, 10, 3., Some(dataset.test_data));

    Ok(())
}
