use opencv::{Result, highgui, imgcodecs};

use computer_vision::histogram::*;

fn main() -> Result<()> {
    let filenames = [
        "A-Virgem-Das-Rochas",
        "La-Nascita-Di-Venere",
        "Mona-Lisa",
        "Resurrection",
    ];

    for filename in filenames.iter() {
        let image = imgcodecs::imread(&format!("assets/{filename}.jpg"), imgcodecs::IMREAD_COLOR)?;

        let (gray, equalized, negative) = gray_scale_image_transformations(&image)?;

        highgui::named_window("Gray Scale", highgui::WINDOW_GUI_NORMAL)?;
        highgui::imshow("Gray Scale", &gray)?;

        highgui::named_window("Gray Equalized", highgui::WINDOW_GUI_NORMAL)?;
        highgui::imshow("Gray Equalized", &equalized)?;

        highgui::named_window("Gray Negative", highgui::WINDOW_GUI_NORMAL)?;
        highgui::imshow("Gray Negative", &negative)?;

        highgui::wait_key(0)?;

        highgui::destroy_window("Gray Scale")?;
        highgui::destroy_window("Gray Equalized")?;
        highgui::destroy_window("Gray Negative")?;

        let (equalized, negative) = colorful_image_transformation(&image)?;

        highgui::named_window("Original", highgui::WINDOW_GUI_NORMAL)?;
        highgui::imshow("Original", &image)?;

        highgui::named_window("Equalized", highgui::WINDOW_GUI_NORMAL)?;
        highgui::imshow("Equalized", &equalized)?;

        highgui::named_window("Negative", highgui::WINDOW_GUI_NORMAL)?;
        highgui::imshow("Negative", &negative)?;

        highgui::wait_key(0)?;

        highgui::destroy_window("Original")?;
        highgui::destroy_window("Equalized")?;
        highgui::destroy_window("Negative")?;
    }

    Ok(())
}
