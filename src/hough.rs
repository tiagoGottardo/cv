use opencv::{
    Error, Result,
    core::{self, AlgorithmHint, Mat, Point, Scalar, Size, Vec3f, Vector},
    imgproc,
    prelude::*,
};

const CANNY_LOW_THRESHOLD: f64 = 50.0;
const CANNY_HIGH_THRESHOLD: f64 = 150.0;
const HOUGH_ACCUMULATOR_THRESHOLD: f64 = 120.0;
const DEFAULT_ELLIPSE_MIN_MAJOR_AXIS: i32 = 20;
const DEFAULT_ELLIPSE_MIN_MINOR_AXIS: i32 = 10;
const DEFAULT_ELLIPSE_MIN_AXIS_RATIO: f64 = 1.45;
const DEFAULT_ELLIPSE_MIN_CONTOUR_POINTS: usize = 40;
const DEFAULT_ELLIPSE_MIN_APPROX_POINTS: usize = 12;
const DEFAULT_ELLIPSE_MIN_AREA_RATIO: f64 = 0.75;
const DEFAULT_ELLIPSE_MAX_AREA_RATIO: f64 = 1.20;
const DEFAULT_ELLIPSE_MAX_AVERAGE_FIT_ERROR: f64 = 0.08;
const DEFAULT_ELLIPSE_MAX_RESULTS: usize = 8;
const DEFAULT_ELLIPSE_MAX_AXIS_CAP: i32 = 260;

#[derive(Clone, Debug)]
pub struct HoughEllipse {
    pub center: Point,
    pub axes: Size,
    pub angle_degrees: f64,
    pub votes: u32,
}

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

pub fn detect_ellipses_with_hough_and_draw_it(image: &Mat) -> Result<Mat> {
    let ellipses = detect_ellipses_with_hough(image)?;
    let mut result = image.clone();

    for ellipse in ellipses {
        imgproc::ellipse(
            &mut result,
            ellipse.center,
            ellipse.axes,
            ellipse.angle_degrees,
            0.0,
            360.0,
            Scalar::new(255.0, 0.0, 0.0, 0.0),
            2,
            imgproc::LINE_AA,
            0,
        )?;
    }

    Ok(result)
}

pub fn detect_ellipses_with_hough(image: &Mat) -> Result<Vec<HoughEllipse>> {
    let max_major_axis = default_max_major_axis(image);
    let max_minor_axis = default_max_minor_axis(image);
    let edges = ellipse_edges(image)?;
    let mut ellipses = ellipse_candidates_from_edges(&edges, max_major_axis, max_minor_axis)?;

    ellipses.sort_by(|left, right| right.votes.cmp(&left.votes));
    let mut selected = Vec::new();

    for ellipse in ellipses {
        if selected
            .iter()
            .all(|candidate| !ellipses_are_similar(candidate, &ellipse))
        {
            selected.push(ellipse);
        }

        if selected.len() == DEFAULT_ELLIPSE_MAX_RESULTS {
            break;
        }
    }

    Ok(selected)
}

fn ellipse_candidates_from_edges(
    edges: &Mat,
    max_major_axis: i32,
    max_minor_axis: i32,
) -> Result<Vec<HoughEllipse>> {
    let mut contours = Vector::<Vector<Point>>::new();
    imgproc::find_contours(
        edges,
        &mut contours,
        imgproc::RETR_LIST,
        imgproc::CHAIN_APPROX_NONE,
        Point::new(0, 0),
    )?;

    let mut ellipses = Vec::new();

    for contour in contours {
        if contour.len() < DEFAULT_ELLIPSE_MIN_CONTOUR_POINTS {
            continue;
        }

        if is_polygon_contour(&contour)? {
            continue;
        }

        let rotated_rect = imgproc::fit_ellipse(&contour)?;
        let mut major_axis = (rotated_rect.size.width / 2.0).round() as i32;
        let mut minor_axis = (rotated_rect.size.height / 2.0).round() as i32;
        let mut angle_degrees = rotated_rect.angle as f64;

        if minor_axis > major_axis {
            std::mem::swap(&mut major_axis, &mut minor_axis);
            angle_degrees = (angle_degrees + 90.0) % 180.0;
        }

        if major_axis < DEFAULT_ELLIPSE_MIN_MAJOR_AXIS
            || major_axis > max_major_axis
            || minor_axis < DEFAULT_ELLIPSE_MIN_MINOR_AXIS
            || minor_axis > max_minor_axis
            || major_axis as f64 / minor_axis as f64 <= DEFAULT_ELLIPSE_MIN_AXIS_RATIO
        {
            continue;
        }

        let contour_area = imgproc::contour_area(&contour, false)?;
        let ellipse_area = std::f64::consts::PI * major_axis as f64 * minor_axis as f64;
        let area_ratio = contour_area / ellipse_area;

        if !(DEFAULT_ELLIPSE_MIN_AREA_RATIO..=DEFAULT_ELLIPSE_MAX_AREA_RATIO).contains(&area_ratio)
        {
            continue;
        }

        let fit_error = average_ellipse_fit_error(
            &contour,
            rotated_rect.center.x as f64,
            rotated_rect.center.y as f64,
            major_axis as f64,
            minor_axis as f64,
            angle_degrees,
        )?;

        if fit_error > DEFAULT_ELLIPSE_MAX_AVERAGE_FIT_ERROR {
            continue;
        }

        ellipses.push(HoughEllipse {
            center: Point::new(
                rotated_rect.center.x.round() as i32,
                rotated_rect.center.y.round() as i32,
            ),
            axes: Size::new(major_axis, minor_axis),
            angle_degrees,
            votes: contour.len() as u32,
        });
    }

    Ok(ellipses)
}

fn is_polygon_contour(contour: &Vector<Point>) -> Result<bool> {
    let perimeter = imgproc::arc_length(contour, true)?;
    let mut approximated = Vector::<Point>::new();

    imgproc::approx_poly_dp(contour, &mut approximated, 0.01 * perimeter, true)?;

    Ok(approximated.len() < DEFAULT_ELLIPSE_MIN_APPROX_POINTS)
}

fn average_ellipse_fit_error(
    contour: &Vector<Point>,
    center_x: f64,
    center_y: f64,
    major_axis: f64,
    minor_axis: f64,
    angle_degrees: f64,
) -> Result<f64> {
    let rotation = angle_degrees.to_radians();
    let cos_rotation = rotation.cos();
    let sin_rotation = rotation.sin();
    let mut total_error = 0.0;

    for point in contour {
        let x = point.x as f64 - center_x;
        let y = point.y as f64 - center_y;
        let normalized_x = (x * cos_rotation + y * sin_rotation) / major_axis;
        let normalized_y = (-x * sin_rotation + y * cos_rotation) / minor_axis;
        let radius = (normalized_x.powi(2) + normalized_y.powi(2)).sqrt();

        total_error += (radius - 1.0).abs();
    }

    Ok(total_error / contour.len() as f64)
}

fn ellipse_edges(image: &Mat) -> Result<Mat> {
    let mut gray_image = Mat::default();

    match image.channels() {
        1 => gray_image = image.clone(),
        3 => imgproc::cvt_color(
            image,
            &mut gray_image,
            imgproc::COLOR_BGR2GRAY,
            0,
            AlgorithmHint::ALGO_HINT_DEFAULT,
        )?,
        4 => imgproc::cvt_color(
            image,
            &mut gray_image,
            imgproc::COLOR_BGRA2GRAY,
            0,
            AlgorithmHint::ALGO_HINT_DEFAULT,
        )?,
        _ => {
            return Err(Error::new(
                core::StsBadArg,
                "image must have 1, 3, or 4 channels".to_string(),
            ));
        }
    }

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

    let mut edges = canny_edges(&blurred)?;

    if image.channels() > 1 {
        let mut channels = Vector::<Mat>::new();
        core::split(image, &mut channels)?;

        for channel in channels {
            let mut blurred_channel = Mat::default();
            imgproc::gaussian_blur(
                &channel,
                &mut blurred_channel,
                Size::new(5, 5),
                1.5,
                1.5,
                core::BORDER_DEFAULT,
                AlgorithmHint::ALGO_HINT_DEFAULT,
            )?;

            let channel_edges = canny_edges(&blurred_channel)?;
            let mut combined_edges = Mat::default();
            core::bitwise_or(
                &edges,
                &channel_edges,
                &mut combined_edges,
                &core::no_array(),
            )?;
            edges = combined_edges;
        }
    }

    Ok(edges)
}

fn canny_edges(image: &Mat) -> Result<Mat> {
    let mut edges = Mat::default();
    imgproc::canny(
        image,
        &mut edges,
        CANNY_LOW_THRESHOLD,
        CANNY_HIGH_THRESHOLD,
        3,
        true,
    )?;

    Ok(edges)
}

fn default_max_major_axis(image: &Mat) -> i32 {
    ((image.cols().max(image.rows()) - 1) / 2)
        .min(DEFAULT_ELLIPSE_MAX_AXIS_CAP)
        .max(DEFAULT_ELLIPSE_MIN_MAJOR_AXIS)
}

fn default_max_minor_axis(image: &Mat) -> i32 {
    ((image.cols().min(image.rows()) - 1) / 2)
        .min(DEFAULT_ELLIPSE_MAX_AXIS_CAP)
        .max(DEFAULT_ELLIPSE_MIN_MINOR_AXIS)
}

fn ellipses_are_similar(left: &HoughEllipse, right: &HoughEllipse) -> bool {
    let center_distance_x = (left.center.x - right.center.x).abs();
    let center_distance_y = (left.center.y - right.center.y).abs();
    let major_axis_distance = (left.axes.width - right.axes.width).abs();
    let minor_axis_distance = (left.axes.height - right.axes.height).abs();
    let angle_distance = (left.angle_degrees - right.angle_degrees).abs();
    let normalized_angle_distance = angle_distance.min(180.0 - angle_distance);

    center_distance_x <= 8
        && center_distance_y <= 8
        && major_axis_distance <= 10
        && minor_axis_distance <= 10
        && normalized_angle_distance <= 15.0
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
        150,
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
