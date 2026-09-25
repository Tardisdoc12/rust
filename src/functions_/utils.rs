// utils.rs

pub fn safe_float(x: &str) -> f32 {
    x.parse::<f32>().unwrap_or(0.0)
}

pub fn clean_double_dot(price: &str) -> String {
    let chars: Vec<char> = price.chars().collect();
    let mut result = String::new();

    for i in 0..chars.len() {
        let is_digit = chars[i].is_ascii_digit();
        let next_is_digit = i + 1 < chars.len() && chars[i + 1].is_ascii_digit();
        if !is_digit && !next_is_digit {
            continue;
        }
        result.push(chars[i]);
    }
    result
}

pub fn clean_thousand_dot(price: &str) -> String {
    let chars: Vec<char> = price.chars().collect();
    let mut reversed: Vec<char> = Vec::new();
    let mut counter_dot = 0;

    for i in (0..chars.len()).rev() {
        if chars[i] == '.' {
            counter_dot += 1;
            if counter_dot > 1 {
                continue;
            }
        }
        reversed.push(chars[i]);
    }
    reversed.reverse();
    let final_price: String = reversed.into_iter().collect();

    let price_parts: Vec<&str> = final_price.split('.').collect();
    if price_parts.len() == 2 {
        let frac = price_parts[1];
        match frac.len() {
            0..=2 => final_price,
            3..=4 => price_parts.concat(),
            5 => format!("{}{}.{}", price_parts[0], &frac[..3], &frac[3..]),
            _ => final_price,
        }
    } else {
        final_price
    }
}