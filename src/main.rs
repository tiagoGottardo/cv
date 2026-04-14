// Quais são os métodos escolhidos?
// - Canny
// - Laplacian
// - Sobel
//
// Quais imagens usaremos?
// - The Virgin of the Rocks – Leonardo da Vinci
// - The Arnolfini Portrait – Jan van Eyck

use opencv::{Result, core::Mat, highgui, imgcodecs, imgproc};

fn canny(filename: &str) -> Result<Mat> {
    let src = imgcodecs::imread(filename, imgcodecs::IMREAD_GRAYSCALE)?;
    let mut edges = Mat::default();

    imgproc::canny(&src, &mut edges, 50.0, 150.0, 3, true)?;

    Ok(edges)
}

use opencv::core::{self, AlgorithmHint};

fn laplacian(filename: &str) -> Result<Mat> {
    let src = imgcodecs::imread(filename, imgcodecs::IMREAD_GRAYSCALE)?;
    let mut blurred = Mat::default();
    let mut laplacian = Mat::default();
    let mut abs_laplacian = Mat::default();

    imgproc::gaussian_blur(
        &src,
        &mut blurred,
        core::Size::new(3, 3),
        0.0,
        0.0,
        core::BORDER_DEFAULT,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    imgproc::laplacian(
        &blurred,
        &mut laplacian,
        core::CV_16S,
        1,   // ksize
        1.0, // scale
        0.0, // delta
        core::BORDER_DEFAULT,
    )?;

    core::convert_scale_abs(&laplacian, &mut abs_laplacian, 1.0, 0.0)?;

    // 3. Convert back to 8-bit for visualization
    let mut normalized = Mat::default();
    let mut binary = Mat::default();

    core::normalize(
        &abs_laplacian,
        &mut normalized,
        0.0,
        255.0,
        core::NORM_MINMAX,
        core::CV_8U,
        &Mat::default(),
    )?;

    // 2. Ou aplicar um threshold para separar as bordas do fundo cinza
    imgproc::threshold(
        &abs_laplacian,
        &mut binary,
        30.0, // Ajuste esse valor conforme necessário
        255.0,
        imgproc::THRESH_BINARY,
    )?;

    Ok(binary)
}

fn sobel(filename: &str) -> Result<Mat> {
    let src = imgcodecs::imread(filename, imgcodecs::IMREAD_GRAYSCALE)?;
    let mut grad_x = Mat::default();
    let mut grad_y = Mat::default();
    let mut abs_grad_x = Mat::default();
    let mut abs_grad_y = Mat::default();
    let mut sobel = Mat::default();

    imgproc::sobel(
        &src,
        &mut grad_x,
        core::CV_16S,
        1,
        0,
        3,
        1.0,
        0.0,
        core::BORDER_DEFAULT,
    )?;
    // Gradient Y
    imgproc::sobel(
        &src,
        &mut grad_y,
        core::CV_16S,
        0,
        1,
        3,
        1.0,
        0.0,
        core::BORDER_DEFAULT,
    )?;

    core::convert_scale_abs(&grad_x, &mut abs_grad_x, 1.0, 0.0)?;
    core::convert_scale_abs(&grad_y, &mut abs_grad_y, 1.0, 0.0)?;

    core::add_weighted(&abs_grad_x, 0.5, &abs_grad_y, 0.5, 0.0, &mut sobel, -1)?;

    Ok(sobel)
}

fn main() -> Result<()> {
    highgui::imshow("Canny", &canny("mona_lisa.jpg")?)?;
    highgui::imshow("Laplacian", &laplacian("mona_lisa.jpg")?)?;
    highgui::imshow("Sobel", &sobel("mona_lisa.jpg")?)?;
    highgui::wait_key(0)?;

    // imgcodecs::imwrite(
    //     "laplacian-mona-lisa2.jpg",
    //     &laplacian("mona_lisa.jpg")?,
    //     &opencv::core::Vector::default(),
    // )?;

    Ok(())
}
