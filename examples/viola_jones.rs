use computer_vision::viola_jones::{
    self, classify, draw_label, generate_confusion_table, get_classifications, parse_label,
    validate_classification, viola_jones,
};
use opencv::{Result, highgui, imgcodecs};

const FOLDER_PATH: &str = "/home/tiagopg/Downloads/dataset2/baseline/highway/input";
const FILE_PATH: &str = "/home/tiagopg/Downloads/dataset2/baseline/highway/input/in000960.jpg";
const MODEL_PATH: &str = "assets/haarcascade_car.xml";

fn main() -> Result<()> {
    // let original_image = imgcodecs::imread(FILE_PATH, imgcodecs::IMREAD_COLOR)?;

    let result_image = viola_jones(FILE_PATH, MODEL_PATH)?;

    let labels = parse_label("assets/cars_label.txt");

    let result_image = draw_label(result_image, &labels[959])?;

    highgui::imshow("Detected Vehicles", &result_image)?;
    highgui::wait_key(0)?;

    // let detected = classify(&original_image, MODEL_PATH)?;

    // let result = validate_classification(detected, &labels[959]);

    let classifications = get_classifications(FOLDER_PATH, MODEL_PATH)?;

    let table = generate_confusion_table(classifications, labels);

    dbg!(table);

    Ok(())
}

// fn main() -> Result<()> {
//     let img = viola_jones("assets/highway.jpg", "assets/haarcascade_car.xml")?;
//
//     highgui::imshow("Detected Vehicles", &img)?;
//     highgui::wait_key(0)?;
//
//     Ok(())
// }
