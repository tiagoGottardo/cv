use opencv::{Result, highgui, imgcodecs};

use computer_vision::edge_detectors::*;

fn main() -> Result<()> {
    let filenames = vec!["The-Arnolfini-Portrait", "Vergine-Delle-Rocce"];

    for filename in filenames.into_iter() {
        let image = imgcodecs::imread(
            &format!("assets/{filename}.jpg"),
            imgcodecs::IMREAD_GRAYSCALE,
        )?;

        let canny_edges = canny(&image)?;
        let laplacian_edges = laplacian(&image)?;
        let sobel_edges = sobel(&image)?;

        highgui::named_window("Gray Scale", highgui::WINDOW_GUI_NORMAL)?;
        highgui::imshow("Gray Scale", &image)?;

        highgui::named_window("Canny", highgui::WINDOW_GUI_NORMAL)?;
        highgui::imshow("Canny", &canny_edges)?;

        highgui::named_window("Laplacian", highgui::WINDOW_GUI_NORMAL)?;
        highgui::imshow("Laplacian", &laplacian_edges)?;

        highgui::named_window("Sobel", highgui::WINDOW_GUI_NORMAL)?;
        highgui::imshow("Sobel", &sobel_edges)?;

        highgui::wait_key(0)?;

        highgui::destroy_window("Gray Scale")?;
        highgui::destroy_window("Canny")?;
        highgui::destroy_window("Laplacian")?;
        highgui::destroy_window("Sobel")?;
    }

    Ok(())
}
