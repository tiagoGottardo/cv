use opencv::{Result, highgui, imgcodecs};

use computer_vision::hough::*;

fn main() -> Result<()> {
    let images = vec!["Circles-In-A-Circle.jpg", "Brinquedos.jpg"];

    for image_path in images.into_iter() {
        let image = imgcodecs::imread(&format!("assets/{image_path}"), imgcodecs::IMREAD_COLOR)?;

        let circle = detect_circles_with_hough_and_draw_it(&image)?;
        let ellipses = detect_ellipses_with_hough_and_draw_it(&image)?;
        let lines = detect_lines_with_hough_and_draw_it(&image)?;

        highgui::named_window("Original", highgui::WINDOW_GUI_NORMAL)?;
        highgui::imshow("Original", &image)?;

        highgui::named_window("Hough Circles", highgui::WINDOW_GUI_NORMAL)?;
        highgui::imshow("Hough Circles", &circle)?;

        highgui::named_window("Hough Ellipses", highgui::WINDOW_GUI_NORMAL)?;
        highgui::imshow("Hough Ellipses", &ellipses)?;

        highgui::named_window("Hough Lines", highgui::WINDOW_GUI_NORMAL)?;
        highgui::imshow("Hough Lines", &lines)?;

        highgui::wait_key(0)?;
    }

    Ok(())
}
