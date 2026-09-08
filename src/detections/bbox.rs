//--------------------------------------------------------------------------------------------------
// Filename: bbox.rs
// Author: Jean Anquetil
// Date: 2026-09-07
//--------------------------------------------------------------------------------------------------

use pyo3::prelude::*;

//--------------------------------------------------------------------------------------------------


#[pyclass(name = "BBox")]
#[derive(Clone)]
pub struct BBox {
    #[pyo3(get, set)]
    pub x1: f32,
    #[pyo3(get, set)]
    pub y1: f32,
    #[pyo3(get, set)]
    pub x2: f32,
    #[pyo3(get, set)]
    pub y2: f32,
    #[pyo3(get, set)]
    pub img_shape: (usize, usize),
}

//--------------------------------------------------------------------------------------------------

#[pymethods]
impl BBox {
    #[new]
    pub fn py_new(
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        img_shape: (usize, usize)
    ) -> Self {
        Self::new(x1, y1, x2, y2, img_shape)
    }
}
impl BBox {
    pub fn new(
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        img_shape: (usize, usize)
    ) -> Self {
        Self {
            x1: x1 / img_shape.1 as f32,
            y1: y1 / img_shape.0 as f32,
            x2: x2 / img_shape.1 as f32,
            y2: y2 / img_shape.0 as f32,
            img_shape
        }
    }

    pub fn area(&self) -> f32 {
        return (self.x2 - self.x1) * (self.y2 - self.y1)
    }

    pub fn width(&self) -> f32 {
        return self.x2 - self.x1;
    }

    pub fn height(&self) -> f32 {
        return self.y2 - self.y1;
    }

    pub fn center_rel(&self) -> (f32, f32) {
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0);
    }

    pub fn center_abs(&self) -> (f32, f32) {
        return (
            (self.x1 + self.x2) / 2.0 * self.img_shape.1 as f32,
            (self.y1 + self.y2) / 2.0 * self.img_shape.0 as f32
        );
    }

    pub fn xyxyn(&self) -> (f32, f32, f32, f32) {
        return (self.x1, self.y1, self.x2, self.y2);
    }

    pub fn xyxy(&self) -> (f32, f32, f32, f32) {
        return (
            self.x1 * self.img_shape.1 as f32,
            self.y1 * self.img_shape.0 as f32,
            self.x2 * self.img_shape.1 as f32,
            self.y2 * self.img_shape.0 as f32
        );
    }
}

//--------------------------------------------------------------------------------------------------
// End of file
//--------------------------------------------------------------------------------------------------