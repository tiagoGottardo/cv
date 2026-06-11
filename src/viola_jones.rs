use std::fs;
use std::io::BufRead;
use std::path::PathBuf;
use std::{fs::File, io::BufReader};

use opencv::{
    Result,
    core::{Rect_, Scalar, Size, Vector},
    imgcodecs, imgproc,
    objdetect::CascadeClassifier,
    prelude::*,
};

const SCALE_FACTOR: f64 = 1.1;
const MIN_NEIGHBORS: i32 = 2;
const MIN_SIZE: Size = Size::new(0, 0);
const MAX_SIZE: Size = Size::new(100, 100);

pub fn classify(img: &Mat, model_path: &str) -> Result<Vector<Rect_<i32>>> {
    let mut gray = Mat::default();
    imgproc::cvt_color(
        img,
        &mut gray,
        imgproc::COLOR_BGR2GRAY,
        0,
        opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    let mut detector = CascadeClassifier::new(model_path)?;

    let mut detected = Vector::<opencv::core::Rect>::new();
    detector.detect_multi_scale(
        &gray,
        &mut detected,
        SCALE_FACTOR,
        MIN_NEIGHBORS,
        0,
        MIN_SIZE,
        MAX_SIZE,
    )?;

    Ok(detected)
}

pub fn viola_jones(image_path: &str, model_path: &str) -> Result<Mat> {
    let mut img = imgcodecs::imread(image_path, imgcodecs::IMREAD_COLOR)?;

    let detected = classify(&img, model_path)?;

    for rect in detected.iter() {
        imgproc::rectangle(
            &mut img,
            rect,
            Scalar::new(0.0, 255.0, 0.0, 0.0),
            2,
            imgproc::LINE_8,
            0,
        )?;
    }

    Ok(img)
}

pub fn parse_label(path: &str) -> Vec<Label> {
    let file = File::open(path).expect("Label file not found!");
    let reader = BufReader::new(file);

    reader
        .lines()
        .map(|line| parse_line(&line.expect("Some went wrong on get line!")))
        // .filter(|label| label.amount > 0)
        .collect::<Vec<Label>>()
}

#[derive(Debug, Default)]
pub struct Label {
    pub image: i32,
    pub amount: i32,
    pub coordinates: Vec<(i32, i32)>,
}

pub fn parse_line(line: &str) -> Label {
    let mut label = Label::default();

    line.split_whitespace()
        .enumerate()
        .for_each(|(i, item)| match i {
            0 => label.image = item.parse::<i32>().expect("Error on parse to i32!"),
            1 => label.amount = item.parse::<i32>().expect("Error on parse to i32!"),
            x if x % 2 == 0 => label
                .coordinates
                .push((item.parse::<i32>().expect("Error on parse to i32!"), 0)),
            _ => {
                label.coordinates.last_mut().expect("Some shit!").1 =
                    item.parse::<i32>().expect("Error on parse to i32!")
            }
        });

    label
}

pub fn draw_label(mut img: Mat, label: &Label) -> Result<Mat> {
    for (x, y) in label.coordinates.iter() {
        imgproc::rectangle(
            &mut img,
            Rect_::new(*x, *y, 2, 2),
            Scalar::new(0., 0., 255., 0.),
            2,
            imgproc::LINE_8,
            0,
        )?;
    }

    Ok(img)
}

const TOLERANCE: i32 = 10;

// (VP FP FN)
pub fn validate_classification(output: Vector<Rect_<i32>>, label: &Label) -> (i32, i32, i32) {
    let mut validation = (0, 0, 0);

    let mut output = output
        .iter()
        .map(|rect| (rect.x + rect.width / 2, rect.y + rect.height / 2))
        .collect::<Vec<(i32, i32)>>();

    for (lx, ly) in label.coordinates.iter() {
        let previous_output_len = output.len();

        for (i, (ox, oy)) in output.iter().enumerate() {
            let diff = ((lx - ox).abs(), (ly - oy).abs());

            if diff.0 < TOLERANCE && diff.1 < TOLERANCE {
                validation.0 += 1;
                output.remove(i);
                break;
            }
        }

        if previous_output_len == output.len() {
            validation.2 += 1;
        }
    }

    validation.1 += output.len() as i32;

    validation
}

pub fn get_classifications(folder_path: &str, model_path: &str) -> Result<Vec<Vector<Rect_<i32>>>> {
    let mut paths: Vec<PathBuf> = fs::read_dir(folder_path)
        .expect("Failed to read directory!")
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| {
            let ext = path
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_lowercase();
            ext == "jpg"
        })
        .collect();

    paths.sort();

    let mut images: Vec<Mat> = Vec::new();

    for path in paths {
        let path_str = path.to_str().expect("Invalid Path!");

        let img = imgcodecs::imread(path_str, imgcodecs::IMREAD_COLOR)?;

        if !img.empty() {
            images.push(img);
        }
    }

    Ok(images
        .into_iter()
        .map(|image| classify(&image, model_path))
        .filter_map(Result::ok)
        .collect::<Vec<Vector<Rect_<i32>>>>())
}

pub fn generate_confusion_table(
    classifications: Vec<Vector<Rect_<i32>>>,
    labels: Vec<Label>,
) -> (i32, i32, i32) {
    let mut result = (0, 0, 0);

    for (i, classification) in classifications.into_iter().enumerate() {
        let validation = validate_classification(classification, &labels[i]);

        result = (
            result.0 + validation.0,
            result.1 + validation.1,
            result.2 + validation.2,
        );
    }

    result
}
