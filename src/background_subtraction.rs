use opencv::{
    Error, Result,
    core::{self, AlgorithmHint, Mat, Point, Rect, Scalar, Size, Vector},
    imgcodecs, imgproc,
    prelude::*,
};
use std::{
    fs,
    path::{Path, PathBuf},
};

const DIFFERENCE_BLUR_SIZE: i32 = 5;
const MASK_MORPH_SIZE: i32 = 7;
const MIN_OBJECT_AREA: i32 = 400;

pub fn get_video_frames_by_folder(folder_path: &str) -> Result<Vec<Mat>> {
    let mut paths = image_paths_from_folder(folder_path)?;
    paths.sort();

    let mut frames = Vec::with_capacity(paths.len());

    for path in paths {
        let path = path.to_string_lossy();
        let frame = imgcodecs::imread(&path, imgcodecs::IMREAD_COLOR)?;

        if frame.empty() {
            return Err(Error::new(
                core::StsError,
                format!("failed to read image frame: {path}"),
            ));
        }

        frames.push(frame);
    }

    if frames.is_empty() {
        return Err(Error::new(
            core::StsBadArg,
            format!("no image frames found in folder: {folder_path}"),
        ));
    }

    Ok(frames)
}

pub fn fixed_background(image: &Mat, background: &Mat, threshold: f64) -> Result<Mat> {
    let image = to_gray(image)?;
    let background = to_gray(background)?;
    ensure_same_size(&image, &background)?;

    let mut difference = Mat::default();
    core::absdiff(&image, &background, &mut difference)?;

    let mut blurred_difference = Mat::default();
    imgproc::gaussian_blur(
        &difference,
        &mut blurred_difference,
        Size::new(DIFFERENCE_BLUR_SIZE, DIFFERENCE_BLUR_SIZE),
        0.0,
        0.0,
        core::BORDER_DEFAULT,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    let mut foreground = Mat::default();
    imgproc::threshold(
        &blurred_difference,
        &mut foreground,
        threshold,
        255.0,
        imgproc::THRESH_BINARY,
    )?;

    clean_foreground_mask(&foreground)
}

fn clean_foreground_mask(mask: &Mat) -> Result<Mat> {
    let kernel = imgproc::get_structuring_element(
        imgproc::MORPH_RECT,
        Size::new(MASK_MORPH_SIZE, MASK_MORPH_SIZE),
        Point::new(-1, -1),
    )?;
    let mut opened = Mat::default();
    imgproc::morphology_ex(
        mask,
        &mut opened,
        imgproc::MORPH_OPEN,
        &kernel,
        Point::new(-1, -1),
        1,
        core::BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?,
    )?;

    let mut closed = Mat::default();
    imgproc::morphology_ex(
        &opened,
        &mut closed,
        imgproc::MORPH_CLOSE,
        &kernel,
        Point::new(-1, -1),
        1,
        core::BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?,
    )?;

    Ok(closed)
}

pub fn mean_filter(image: &Mat, background_images: &[Mat], threshold: f64) -> Result<Mat> {
    let background = mean_background(background_images)?;

    fixed_background(image, &background, threshold)
}

pub fn median_filter(image: &Mat, background_images: &[Mat], threshold: f64) -> Result<Mat> {
    let background = median_background(background_images)?;

    fixed_background(image, &background, threshold)
}

pub fn draw_objects_on_image(original_image: &Mat, foreground_mask: &Mat) -> Result<Mat> {
    let mask = to_gray(foreground_mask)?;
    ensure_same_size(original_image, &mask)?;

    let mut contours = Vector::<Vector<Point>>::new();
    imgproc::find_contours(
        &mask,
        &mut contours,
        imgproc::RETR_EXTERNAL,
        imgproc::CHAIN_APPROX_SIMPLE,
        Point::new(0, 0),
    )?;

    let mut rectangles = Vec::new();

    for contour in contours {
        let rectangle = imgproc::bounding_rect(&contour)?;

        if rectangle.area() >= MIN_OBJECT_AREA {
            rectangles.push(rectangle);
        }
    }

    rectangles.sort_by_key(|rectangle| (rectangle.y, rectangle.x));

    let mut result = image_to_bgr(original_image)?;

    for (index, rectangle) in rectangles.iter().enumerate() {
        let object_name = format!("Obj{}", index + 1);
        draw_object_rectangle(&mut result, *rectangle, &object_name)?;
    }

    Ok(result)
}

fn mean_background(images: &[Mat]) -> Result<Mat> {
    let first = first_gray_image(images)?;
    let mut accumulator = Mat::zeros(first.rows(), first.cols(), core::CV_32FC1)?.to_mat()?;

    for image in images {
        let gray = to_gray(image)?;
        ensure_same_size(&gray, &first)?;

        let mut gray_float = Mat::default();
        gray.convert_to(&mut gray_float, core::CV_32F, 1.0, 0.0)?;

        let mut next_accumulator = Mat::default();
        core::add(
            &accumulator,
            &gray_float,
            &mut next_accumulator,
            &core::no_array(),
            -1,
        )?;
        accumulator = next_accumulator;
    }

    let mut background = Mat::default();
    accumulator.convert_to(&mut background, core::CV_8U, 1.0 / images.len() as f64, 0.0)?;

    Ok(background)
}

fn image_to_bgr(image: &Mat) -> Result<Mat> {
    let mut result = Mat::default();

    match image.channels() {
        1 => imgproc::cvt_color(
            image,
            &mut result,
            imgproc::COLOR_GRAY2BGR,
            0,
            AlgorithmHint::ALGO_HINT_DEFAULT,
        )?,
        3 => result = image.clone(),
        4 => imgproc::cvt_color(
            image,
            &mut result,
            imgproc::COLOR_BGRA2BGR,
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

    Ok(result)
}

fn draw_object_rectangle(image: &mut Mat, rectangle: Rect, object_name: &str) -> Result<()> {
    let color = Scalar::new(0.0, 255.0, 0.0, 0.0);
    imgproc::rectangle(image, rectangle, color, 2, imgproc::LINE_8, 0)?;

    let label_position = Point::new(rectangle.x, (rectangle.y - 8).max(16));
    imgproc::put_text(
        image,
        object_name,
        label_position,
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        imgproc::LINE_AA,
        false,
    )?;

    Ok(())
}

fn median_background(images: &[Mat]) -> Result<Mat> {
    let first = first_gray_image(images)?;
    let rows = first.rows();
    let cols = first.cols();
    let mut gray_images = Vec::with_capacity(images.len());

    for image in images {
        let gray = to_gray(image)?;
        ensure_same_size(&gray, &first)?;
        gray_images.push(gray);
    }

    let mut background =
        Mat::new_rows_cols_with_default(rows, cols, core::CV_8UC1, Scalar::all(0.0))?;
    let median_index = gray_images.len() / 2;
    let mut values = vec![0_u8; gray_images.len()];

    for row in 0..rows {
        for col in 0..cols {
            for (index, image) in gray_images.iter().enumerate() {
                values[index] = *image.at_2d::<u8>(row, col)?;
            }

            values.sort_unstable();
            *background.at_2d_mut::<u8>(row, col)? = values[median_index];
        }
    }

    Ok(background)
}

fn first_gray_image(images: &[Mat]) -> Result<Mat> {
    if images.is_empty() {
        return Err(Error::new(
            core::StsBadArg,
            "background_images must not be empty".to_string(),
        ));
    }

    to_gray(&images[0])
}

fn to_gray(image: &Mat) -> Result<Mat> {
    let mut gray = Mat::default();

    match image.channels() {
        1 => gray = image.clone(),
        3 => imgproc::cvt_color(
            image,
            &mut gray,
            imgproc::COLOR_BGR2GRAY,
            0,
            AlgorithmHint::ALGO_HINT_DEFAULT,
        )?,
        4 => imgproc::cvt_color(
            image,
            &mut gray,
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

    Ok(gray)
}

fn ensure_same_size(image: &Mat, background: &Mat) -> Result<()> {
    if image.size()? != background.size()? {
        return Err(Error::new(
            core::StsUnmatchedSizes,
            format!(
                "all images must have the same size: got {:?} and {:?}",
                image.size()?,
                background.size()?
            ),
        ));
    }

    Ok(())
}

fn image_paths_from_folder(folder_path: &str) -> Result<Vec<PathBuf>> {
    let entries = fs::read_dir(folder_path).map_err(|error| {
        Error::new(
            core::StsError,
            format!("failed to read folder {folder_path}: {error}"),
        )
    })?;
    let mut paths = Vec::new();

    for entry in entries {
        let entry = entry.map_err(|error| {
            Error::new(
                core::StsError,
                format!("failed to read folder entry in {folder_path}: {error}"),
            )
        })?;
        let path = entry.path();

        if path.is_file() && is_image_file(&path) {
            paths.push(path);
        }
    }

    Ok(paths)
}

fn is_image_file(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "bmp" | "jpeg" | "jpg" | "png" | "tif" | "tiff" | "webp"
            )
        })
        .unwrap_or(false)
}
