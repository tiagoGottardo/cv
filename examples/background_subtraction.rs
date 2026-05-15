use computer_vision::background_subtraction::*;
use opencv::{Result, core::Mat, highgui};

fn main() -> Result<()> {
    let frames = get_video_frames_by_folder("assets/vehicles")?;
    let image = frames.get(34).expect("Too few frames");

    let fixed = fixed_background(&image, &frames.get(0).expect("Too few frames"), 30.)?;
    let mean = mean_filter(&image, &frames, 30.0)?;
    let median = median_filter(&image, &frames, 30.0)?;

    see("Original", &image)?;
    see("Fixed", &fixed)?;
    see("Mean Filter", &mean)?;
    see("Median Filter", &median)?;

    highgui::wait_key(0)?;

    Ok(())
}

fn see(title: &str, img: &Mat) -> Result<()> {
    highgui::named_window(title, highgui::WINDOW_GUI_NORMAL)?;
    highgui::imshow(title, &img)?;

    Ok(())
}
