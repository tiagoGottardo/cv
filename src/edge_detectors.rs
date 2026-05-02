use opencv::{
    Result,
    core::{self, AlgorithmHint, Mat},
    imgproc::{self},
    prelude::*,
};

const CANNY_LOW_THRESHOLD: f64 = 50.0;
const CANNY_HIGH_THRESHOLD: f64 = 150.0;

pub fn canny(src: &Mat) -> Result<Mat> {
    let mut blurred = Mat::default();
    imgproc::gaussian_blur(
        src,
        &mut blurred,
        core::Size::new(3, 3),
        0.0,
        0.0,
        core::BORDER_DEFAULT,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    let mut edges = Mat::default();
    imgproc::canny(
        &blurred,
        &mut edges,
        CANNY_LOW_THRESHOLD,
        CANNY_HIGH_THRESHOLD,
        3,
        true,
    )?;

    Ok(edges)
}

pub fn laplacian(src: &Mat) -> Result<Mat> {
    let mut blurred = Mat::default();
    imgproc::gaussian_blur(
        src,
        &mut blurred,
        core::Size::new(3, 3),
        0.0,
        0.0,
        core::BORDER_DEFAULT,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

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

pub fn sobel(image: &Mat) -> Result<Mat> {
    let mut blurred = Mat::default();
    imgproc::gaussian_blur(
        image,
        &mut blurred,
        core::Size::new(3, 3),
        0.0,
        0.0,
        core::BORDER_DEFAULT,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    let mut grad_x = Mat::default();
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

    let mut grad_y = Mat::default();
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

    let mut grad_x_float = Mat::default();
    grad_x.convert_to(&mut grad_x_float, core::CV_32F, 1.0, 0.0)?;

    let mut grad_y_float = Mat::default();
    grad_y.convert_to(&mut grad_y_float, core::CV_32F, 1.0, 0.0)?;

    let mut magnitude = Mat::default();
    core::magnitude(&grad_x_float, &grad_y_float, &mut magnitude)?;

    let mut magnitude_u8 = Mat::default();
    core::normalize(
        &magnitude,
        &mut magnitude_u8,
        0.0,
        255.0,
        core::NORM_MINMAX,
        core::CV_8U,
        &Mat::default(),
    )?;

    let mut edges = Mat::default();
    imgproc::threshold(
        &magnitude_u8,
        &mut edges,
        0.0,
        255.0,
        imgproc::THRESH_BINARY | imgproc::THRESH_OTSU,
    )?;

    Ok(edges)
}
