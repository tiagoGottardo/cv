use opencv::{
    Error, Result,
    core::{self, AlgorithmHint, Mat},
    highgui, imgcodecs, imgproc,
    prelude::*,
};

const INPUT_IMAGES: [&str; 2] = ["The-Arnolfini-portrait", "Vergine-delle-Rocce"];
const CANNY_LOW_THRESHOLD: f64 = 50.0;
const CANNY_HIGH_THRESHOLD: f64 = 150.0;

fn load_grayscale(filename: &str) -> Result<Mat> {
    let src = imgcodecs::imread(filename, imgcodecs::IMREAD_GRAYSCALE)?;
    if src.empty() {
        return Err(Error::new(
            core::StsError,
            format!("could not read image: {filename}"),
        ));
    }

    Ok(src)
}

fn blur(src: &Mat) -> Result<Mat> {
    blur_with_kernel(src, 3)
}

fn blur_with_kernel(src: &Mat, kernel_size: i32) -> Result<Mat> {
    let mut blurred = Mat::default();
    imgproc::gaussian_blur(
        src,
        &mut blurred,
        core::Size::new(kernel_size, kernel_size),
        0.0,
        0.0,
        core::BORDER_DEFAULT,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    Ok(blurred)
}

fn canny(src: &Mat, low_threshold: f64, high_threshold: f64) -> Result<Mat> {
    let blurred = blur(src)?;
    let mut edges = Mat::default();

    imgproc::canny(&blurred, &mut edges, low_threshold, high_threshold, 3, true)?;

    Ok(edges)
}

fn laplacian(src: &Mat) -> Result<Mat> {
    let blurred = blur(src)?;
    let mut laplacian = Mat::default();
    let mut abs_laplacian = Mat::default();
    let mut binary = Mat::default();

    imgproc::laplacian(
        &blurred,
        &mut laplacian,
        core::CV_16S,
        1,
        1.0,
        0.0,
        core::BORDER_DEFAULT,
    )?;

    core::convert_scale_abs(&laplacian, &mut abs_laplacian, 1.0, 0.0)?;
    imgproc::threshold(
        &abs_laplacian,
        &mut binary,
        0.0,
        255.0,
        imgproc::THRESH_BINARY | imgproc::THRESH_OTSU,
    )?;

    Ok(binary)
}

fn sobel(src: &Mat) -> Result<Mat> {
    let blurred = blur_with_kernel(src, 5)?;
    let mut grad_x = Mat::default();
    let mut grad_y = Mat::default();
    let mut grad_x_float = Mat::default();
    let mut grad_y_float = Mat::default();
    let mut magnitude = Mat::default();
    let mut magnitude_u8 = Mat::default();
    let mut edges = Mat::default();

    imgproc::sobel(
        &blurred,
        &mut grad_x,
        core::CV_16S,
        1,
        0,
        3,
        1.0,
        0.0,
        core::BORDER_DEFAULT,
    )?;

    imgproc::sobel(
        &blurred,
        &mut grad_y,
        core::CV_16S,
        0,
        1,
        3,
        1.0,
        0.0,
        core::BORDER_DEFAULT,
    )?;

    grad_x.convert_to(&mut grad_x_float, core::CV_32F, 1.0, 0.0)?;
    grad_y.convert_to(&mut grad_y_float, core::CV_32F, 1.0, 0.0)?;
    core::magnitude(&grad_x_float, &grad_y_float, &mut magnitude)?;

    core::normalize(
        &magnitude,
        &mut magnitude_u8,
        0.0,
        255.0,
        core::NORM_MINMAX,
        core::CV_8U,
        &Mat::default(),
    )?;
    imgproc::threshold(
        &magnitude_u8,
        &mut edges,
        0.0,
        255.0,
        imgproc::THRESH_BINARY | imgproc::THRESH_OTSU,
    )?;

    Ok(edges)
}

fn main() -> Result<()> {
    // for input_image in INPUT_IMAGES.into_iter() {
    //     let src = load_grayscale(&format!("{input_image}.jpg"))?;
    //
    //     let canny_edges = canny(&src, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)?;
    //     let laplacian_edges = laplacian(&src)?;
    //     let sobel_edges = sobel(&src)?;
    //
    //     imgcodecs::imwrite(
    //         &format!("Canny-{input_image}.jpg"),
    //         &canny_edges,
    //         &opencv::core::Vector::default(),
    //     )?;
    //     imgcodecs::imwrite(
    //         &format!("Laplacian-{input_image}.jpg"),
    //         &laplacian_edges,
    //         &opencv::core::Vector::default(),
    //     )?;
    //     imgcodecs::imwrite(
    //         &format!("Sobel-{input_image}.jpg"),
    //         &sobel_edges,
    //         &opencv::core::Vector::default(),
    //     )?;
    //
    //     highgui::imshow("Canny", &canny_edges)?;
    //     highgui::imshow("Laplacian", &laplacian_edges)?;
    //     highgui::imshow("Sobel", &sobel_edges)?;
    //     highgui::wait_key(0)?;
    // }

    Ok(())
}
