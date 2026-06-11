use computer_vision::viola_jones::viola_jones;
use opencv::{Result, highgui};

fn main() -> Result<()> {
    let img = viola_jones("assets/highway.jpg", "assets/haarcascade_car.xml")?;

    highgui::imshow("Detected Vehicles", &img)?;
    highgui::wait_key(0)?;

    Ok(())
}
