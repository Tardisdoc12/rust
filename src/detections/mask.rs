use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use numpy::{PyReadonlyArray3, PyArray3, IntoPyArray};
use opencv::core::{Mat, Point, Point2f, Vector, CV_8UC1, CV_8UC3, bitwise_and, Rect};
use opencv::prelude::*;
use opencv::imgproc;

#[pyclass(name = "Mask", unsendable)]
#[derive(Clone)]
pub struct Mask {
    pub mat: Mat,
    pub _mat_bin : Mat,
}


impl Mask {

    pub fn setup_mat(&mut self, mat : Mat) {
        self.mat = mat.try_clone().unwrap();
    }

    /// Construit un Mat binaire (CV_8UC1) à partir des scores bruts renvoyés par SAM2.
    pub fn mat_from_scores(scores: &[f32], width: i32, height: i32) -> anyhow::Result<Mat> {
        let binaire: Vec<u8> = scores
            .iter()
            .map(|&s| if s > 0.0 { 255u8 } else { 0u8 })
            .collect();

        let mat = unsafe {
            Mat::new_rows_cols_with_data_unsafe(
                height,
                width,
                CV_8UC1,
                binaire.as_ptr() as *mut std::ffi::c_void,
                opencv::core::Mat_AUTO_STEP,
            )
        }?;

        // clone obligatoire : `binaire` est droppé à la fin du scope,
        // Mat::new_rows_cols_with_data_unsafe ne fait que référencer le buffer
        Ok(mat.try_clone()?)
    }

    pub fn get_real_size_from_homography(&self, h: &Mat) -> anyhow::Result<(f32, f32)> {
        let corners_opt = self.get_reference_corners_mat()?;
        let corners = match corners_opt {
            Some(c) => c,
            None => return Ok((0.0, 0.0)),
        };

        let transformed_corners = Self::perspective_transform_mat(&corners, h)?;
        let points_cv: opencv::core::Vector<opencv::core::Point2f> = transformed_corners
            .iter()
            .map(|&(x, y)| opencv::core::Point2f::new(x, y))
            .collect();

        let rect = imgproc::min_area_rect(&points_cv)?;
        let size = rect.size;
        Ok((size.width as f32, size.height as f32))
    }

    pub fn perspective_transform_mat(
        points: &[(f32, f32)],
        h: &Mat,
    ) -> anyhow::Result<Vec<(f32, f32)>> {
        // Convertir la liste de points en Vector<Point2f>, format attendu par perspectiveTransform
        let points_cv: Vector<Point2f> = points
            .iter()
            .map(|&(x, y)| Point2f::new(x, y))
            .collect();

        let mut resultat: Vector<Point2f> = Vector::new();

        opencv::core::perspective_transform(&points_cv, &mut resultat, h)?;

        let points_transformes: Vec<(f32, f32)> = resultat
            .iter()
            .map(|p| (p.x, p.y))
            .collect();

        Ok(points_transformes)
    }

    pub fn get_reference_corners_mat(&self) -> anyhow::Result<Option<Vec<(f32, f32)>>> {
        let contour_opt = self.get_global_contour_points()?;

        let contour: Vector<Point> = match contour_opt {
            Some(c) => c,
            None => return Ok(None),
        };

        // --- Essai 1 : approxPolyDP avec epsilon progressif ---
        let perimetre = imgproc::arc_length(&contour, true)?;

        for eps_factor in [0.02, 0.04, 0.06, 0.10] {
            let epsilon = eps_factor * perimetre;

            let mut approx: Vector<Point> = Vector::new();
            imgproc::approx_poly_dp(&contour, &mut approx, epsilon, true)?;

            if approx.len() == 4 {
                let points: Vec<(f32, f32)> = approx
                    .iter()
                    .map(|p| (p.x as f32, p.y as f32))
                    .collect();
                return Ok(Some(points));
            }
        }

        // --- Fallback : minAreaRect -> toujours 4 coins ---
        let rect = imgproc::min_area_rect(&contour)?;

        let mut box_points: Vector<Point2f> = Vector::new();
        imgproc::box_points(rect, &mut box_points)?;

        let points: Vec<(f32, f32)> = box_points
            .iter()
            .map(|p| (p.x, p.y))
            .collect();

        Ok(Some(points))
    }

    /// Version interne de get_global_contour qui renvoie directement le Vector<Point>
    /// (pas encore converti en tuples), pour être réutilisée par d'autres fonctions.
    pub fn get_global_contour_points(&self) -> anyhow::Result<Option<Vector<Point>>> {
        let mut contours: Vector<Vector<Point>> = Vector::new();

        imgproc::find_contours(
            &self._mat_bin,
            &mut contours,
            imgproc::RETR_EXTERNAL,
            imgproc::CHAIN_APPROX_SIMPLE,
            Point::new(0, 0),
        )?;

        if contours.is_empty() {
            return Ok(None);
        }

        let mut meilleur_index = 0;
        let mut meilleure_aire = f64::MIN;

        for i in 0..contours.len() {
            let contour = contours.get(i)?;
            let aire = imgproc::contour_area(&contour, false)?;
            if aire > meilleure_aire {
                meilleure_aire = aire;
                meilleur_index = i;
            }
        }

        Ok(Some(contours.get(meilleur_index)?))
    }

    /// Trouve le contour de plus grande surface dans un masque binaire (1 canal, 8-bit)
    pub fn get_global_contour_mat(&self) -> anyhow::Result<Option<Vec<(f32, f32)>>> {
        let mut contours: Vector<Vector<Point>> = Vector::new();

        imgproc::find_contours(
            &self._mat_bin,
            &mut contours,
            imgproc::RETR_EXTERNAL,
            imgproc::CHAIN_APPROX_SIMPLE,
            Point::new(0, 0),
        )?;

        if contours.is_empty() {
            return Ok(None);
        }

        // Équivalent de max(contours, key=cv2.contourArea)
        let mut meilleur_index = 0;
        let mut meilleure_aire = f64::MIN;

        for i in 0..contours.len() {
            let contour = contours.get(i)?;
            let aire = imgproc::contour_area(&contour, false)?;

            if aire > meilleure_aire {
                meilleure_aire = aire;
                meilleur_index = i;
            }
        }

        let meilleur_contour = contours.get(meilleur_index)?;

        let points: Vec<(f32, f32)> = meilleur_contour
            .iter()
            .map(|p| (p.x as f32, p.y as f32))
            .collect();

        Ok(Some(points))
    }

    pub fn get_subpart_mat(&self, bbox: (i32, i32, i32, i32)) -> anyhow::Result<Mat> {
        let (x1, y1, x2, y2) = bbox;
        let largeur_image = self.mat.cols();
        let hauteur_image = self.mat.rows();

        let x1 = x1.max(0);
        let y1 = y1.max(0);
        let x2 = x2.min(largeur_image);
        let y2 = y2.min(hauteur_image);

        let rect = Rect::new(x1, y1, (x2 - x1).max(1), (y2 - y1).max(1));
        let roi = Mat::roi(&self.mat, rect)?;
        let crop = roi.try_clone()?;

        Ok(crop)
    }

   pub fn apply_binary_mask(&mut self, masque_mat: &Mat) -> anyhow::Result<Mat> {
        let mut resultat = Mat::default();
        bitwise_and(&self.mat, &self.mat, &mut resultat, masque_mat)?;

        self.mat = resultat.try_clone()?;
        self._mat_bin = masque_mat.try_clone()?;

        Ok(resultat)
    }
}

// ------------------------------------------------------------------
// Méthodes exposées à Python : chaînent les fonctions Rust, convertissent
// en numpy UNE SEULE FOIS à la toute fin
// ------------------------------------------------------------------

#[pymethods]
impl Mask {
    #[new]
    pub fn new() -> Self {
        Self { mat: Mat::default(), _mat_bin: Mat::default() }
    }

    pub fn setup_from_python(&mut self, image: PyReadonlyArray3<'_, u8>) -> PyResult<()> {
        let vue = image.as_array();
        let shape = vue.shape();
        let (hauteur, largeur, canaux) = (shape[0], shape[1], shape[2]);

        if canaux != 3 {
            return Err(PyRuntimeError::new_err("L'image doit avoir 3 canaux"));
        }

        let donnees: Vec<u8> = vue.iter().copied().collect();

        let mat = unsafe {
            Mat::new_rows_cols_with_data_unsafe(
                hauteur as i32,
                largeur as i32,
                CV_8UC3,
                donnees.as_ptr() as *mut std::ffi::c_void,
                opencv::core::Mat_AUTO_STEP,
            )
        }
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        self.mat = mat.try_clone().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(())
    }

    /// Version exposée : crop, renvoie directement en numpy
    fn get_subpart<'py>(&self, py: Python<'py>, bbox: (i32, i32, i32, i32)) -> PyResult<Bound<'py, PyArray3<u8>>> {
        let crop = self.get_subpart_mat(bbox)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        mat_vers_numpy(py, &crop)
    }

    /// Pipeline complet : crop PUIS masque, une seule conversion numpy à la fin
    fn get_subpart_masked<'py>(
        &mut self,
        py: Python<'py>,
        bbox: (i32, i32, i32, i32),
        resultat_sam2: &Mask,
    ) -> PyResult<Bound<'py, PyArray3<u8>>> {
        // Étape 1 : crop (reste en Mat, pas de conversion)
        let crop = self.get_subpart_mat(bbox)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        self.mat = crop.try_clone().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let masque_mat = self._mat_bin.try_clone()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let masque = self.apply_binary_mask(&masque_mat)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        // Étape 3 : SEULE conversion numpy, à la toute fin
        mat_vers_numpy(py, &masque)
    }

    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<u8>>> {
        mat_vers_numpy(py, &self.mat)
    }

    
}

fn mat_vers_numpy<'py>(py: Python<'py>, mat: &Mat) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let hauteur = mat.rows() as usize;
    let largeur = mat.cols() as usize;
    let canaux = mat.channels() as usize;

    let data: &[u8] = mat.data_bytes().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let vec_data = data.to_vec();

    let array = ndarray::Array3::from_shape_vec((hauteur, largeur, canaux), vec_data)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    Ok(array.into_pyarray(py))
}