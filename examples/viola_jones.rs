use opencv::{
    Result,
    core::{Scalar, Size, Vector},
    highgui, imgcodecs, imgproc,
    objdetect::CascadeClassifier,
    prelude::*,
};

fn main() -> Result<()> {
    let image_path = "assets/highway.jpg";
    let mut img = imgcodecs::imread(image_path, imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        panic!("Failed to load image: {}", image_path);
    }

    let mut gray = Mat::default();
    imgproc::cvt_color(
        &img,
        &mut gray,
        imgproc::COLOR_BGR2GRAY,
        0,
        opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    let mut car_detector = CascadeClassifier::new("assets/haarcascade_car.xml")?;
    if car_detector.empty()? {
        panic!("Failed to load classifier!");
    }

    let mut detected_cars = Vector::<opencv::core::Rect>::new();
    car_detector.detect_multi_scale(
        &gray,
        &mut detected_cars,
        1.01,
        7,
        0,
        Size::new(0, 0),
        Size::new(100, 100),
    )?;

    println!("Detected Vehicles: {}", detected_cars.len());

    for rect in detected_cars.iter() {
        imgproc::rectangle(
            &mut img,
            rect,
            Scalar::new(0.0, 255.0, 0.0, 0.0),
            2,
            imgproc::LINE_8,
            0,
        )?;
    }

    highgui::imshow("Detected Vehicles", &img)?;
    highgui::wait_key(0)?;

    Ok(())
}
