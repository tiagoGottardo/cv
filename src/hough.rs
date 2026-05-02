use opencv::{
    Result,
    core::{self, AlgorithmHint, Mat, Point, Scalar, Size, Vec3f, Vector},
    imgproc,
};

const CANNY_LOW_THRESHOLD: f64 = 50.0;
const CANNY_HIGH_THRESHOLD: f64 = 150.0;
const HOUGH_ACCUMULATOR_THRESHOLD: f64 = 120.0;

pub fn detect_circles_with_hough_and_draw_it(image: &Mat) -> Result<Mat> {
    let mut gray_image = core::Mat::default();
    imgproc::cvt_color(
        &image,
        &mut gray_image,
        imgproc::COLOR_BGR2GRAY,
        0,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    let mut blurred = Mat::default();

    imgproc::gaussian_blur(
        &gray_image,
        &mut blurred,
        Size::new(5, 5),
        1.5,
        1.5,
        core::BORDER_DEFAULT,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    let mut circles = Vector::<Vec3f>::new();
    imgproc::hough_circles(
        &blurred,
        &mut circles,
        imgproc::HOUGH_GRADIENT,
        1.0,
        13.,
        CANNY_HIGH_THRESHOLD,
        HOUGH_ACCUMULATOR_THRESHOLD,
        0,
        0,
    )?;

    let mut result = image.clone();

    for circle in circles {
        let center = Point::new(circle[0].round() as i32, circle[1].round() as i32);
        let radius = circle[2].round() as i32;

        imgproc::circle(
            &mut result,
            center,
            3,
            core::Scalar::new(0.0, 255.0, 0.0, 0.0),
            -1,
            imgproc::LINE_8,
            0,
        )?;
        imgproc::circle(
            &mut result,
            center,
            radius,
            core::Scalar::new(0.0, 0.0, 255.0, 0.0),
            2,
            imgproc::LINE_8,
            0,
        )?;
    }

    Ok(result)
}

pub fn detect_lines_with_hough_and_draw_it(image: &Mat) -> Result<Mat> {
    let mut gray_image = core::Mat::default();
    imgproc::cvt_color(
        &image,
        &mut gray_image,
        imgproc::COLOR_BGR2GRAY,
        0,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    let mut blurred = Mat::default();
    imgproc::gaussian_blur(
        &gray_image,
        &mut blurred,
        Size::new(5, 5),
        1.5,
        1.5,
        core::BORDER_DEFAULT,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    let mut binary = Mat::default();
    imgproc::canny(
        &blurred,
        &mut binary,
        CANNY_LOW_THRESHOLD,
        HOUGH_ACCUMULATOR_THRESHOLD,
        3,
        true,
    )?;

    let mut lines = core::Vector::<core::Vec2f>::new();
    imgproc::hough_lines(
        &binary,
        &mut lines,
        1.0,
        std::f64::consts::PI / 180.0,
        300,
        0.0,
        0.0,
        0.0,
        std::f64::consts::PI,
        false,
    )?;

    let mut result = image.clone();

    for line in lines.iter() {
        let rho = line[0];
        let theta = line[1];

        let a = theta.cos();
        let b = theta.sin();
        let x0 = a * rho;
        let y0 = b * rho;

        let p1 = Point::new((x0 + 1000.0 * (-b)) as i32, (y0 + 1000.0 * (a)) as i32);
        let p2 = Point::new((x0 - 1000.0 * (-b)) as i32, (y0 - 1000.0 * (a)) as i32);

        imgproc::line(
            &mut result,
            p1,
            p2,
            Scalar::new(0.0, 10.0, 255.0, 0.0),
            2,
            imgproc::LINE_AA,
            0,
        )?;
    }

    Ok(result)
}
