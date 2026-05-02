use opencv::{Result, core::*, imgproc};

pub fn gray_scale_image_transformations(image: &Mat) -> Result<(Mat, Mat, Mat)> {
    let mut gray_image = Mat::default();
    imgproc::cvt_color(
        &image,
        &mut gray_image,
        imgproc::COLOR_BGR2GRAY,
        0,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    let mut negative = Mat::default();
    bitwise_not(&gray_image, &mut negative, &no_array())?;

    let mut equalized = Mat::default();
    imgproc::equalize_hist(&gray_image, &mut equalized)?;

    Ok((gray_image, equalized, negative))
}

pub fn colorful_image_transformation(image: &Mat) -> Result<(Mat, Mat)> {
    let mut negative = Mat::default();
    bitwise_not(&image, &mut negative, &no_array())?;

    let mut ycrcb = Mat::default();
    imgproc::cvt_color(
        &image,
        &mut ycrcb,
        imgproc::COLOR_BGR2YCrCb,
        0,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    let mut channels = Vector::<Mat>::new();
    split(&ycrcb, &mut channels)?;

    let mut equalized_y = Mat::default();
    imgproc::equalize_hist(&channels.get(0)?, &mut equalized_y)?;
    channels.set(0, equalized_y)?;

    let mut result_ycrcb = Mat::default();
    merge(&channels, &mut result_ycrcb)?;

    let mut final_bgr = Mat::default();
    imgproc::cvt_color(
        &result_ycrcb,
        &mut final_bgr,
        imgproc::COLOR_YCrCb2BGR,
        0,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    Ok((final_bgr, negative))
}
