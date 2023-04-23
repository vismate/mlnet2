mod buffer_wrapper;
use buffer_wrapper::BufferWrapper;

use minifb::{Key, Window, WindowOptions};

use dfdx::{
    losses::mse_loss,
    nn::builders::{DeviceBuildExt, Linear, ModuleMut, Sigmoid, ZeroGrads},
    optim::{Adam, AdamConfig, Optimizer},
    prelude::{Module, ReLU},
    shapes::Rank2,
    tensor::{AsArray, Cpu, Tensor, TensorFrom, Trace},
    tensor_ops::Backward,
};

use rand::{distributions::Standard, Rng};

use plotters::{chart::ChartState, coord::types::RangedCoordf32, prelude::*};

use plotters_bitmap::bitmap_pixel::BGRXPixel;
use plotters_bitmap::BitMapBackend;

type Mlp = (
    (Linear<2, 64>, ReLU),
    (Linear<64, 64>, ReLU),
    (Linear<64, 64>, ReLU),
    (Linear<64, 1>, Sigmoid),
);

const W: usize = 800;
const H: usize = 800;

const NSAMPLES: usize = 2000;
const NPLOTSAMPLES: u32 = 160;
const PLOTSIZE: u32 = W as u32;
const COLORMAP: colorous::Gradient = colorous::VIRIDIS;
//////////////////////////////////
const NPLOTSAMPLESSQR: usize = (NPLOTSAMPLES * NPLOTSAMPLES) as usize;
const PW: f32 = 1.0 / NPLOTSAMPLES as f32;

fn generate_data(num_points: usize) -> (Vec<[f32; 2]>, Vec<f32>) {
    let mut rng = rand::thread_rng();

    let mut xs = Vec::with_capacity(num_points);
    let mut ys = Vec::with_capacity(num_points);

    for _ in 0..num_points {
        let point @ [x, y]: [f32; 2] = [rng.sample(Standard), rng.sample(Standard)];
        let dist = (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5);
        let label = if dist > 0.15 * 0.15 && dist < 0.35 * 0.35
            || (x * std::f32::consts::PI * 2.0).cos().abs() * 0.3 > y
        {
            1.0
        } else {
            0.0
        };

        xs.push(point);
        ys.push(label);
    }

    (xs, ys)
}

fn get_meshgrid(dev: &Cpu) -> Tensor<Rank2<NPLOTSAMPLESSQR, 2>, f32, Cpu> {
    let mut meshgrid = Vec::with_capacity(NPLOTSAMPLESSQR * 2);

    for x in 0..NPLOTSAMPLES {
        for y in 0..NPLOTSAMPLES {
            meshgrid.push(x as f32 / NPLOTSAMPLES as f32);
            meshgrid.push(y as f32 / NPLOTSAMPLES as f32);
        }
    }

    dev.tensor(meshgrid)
}

fn graph_decision_map(pred: &[[f32; 1]], xs: &[[f32; 2]], ys: &[f32], buf: &mut [u8]) {
    let mut root = BitMapBackend::<BGRXPixel>::with_buffer_and_format(buf, (W as u32, H as u32))
        .unwrap()
        .into_drawing_area();

    root.fill(&WHITE).unwrap();

    root = root
        .titled("Decision Map", ("sans-serif", 32).into_font().color(&BLACK))
        .unwrap();

    let mut chart = ChartBuilder::on(&root)
        .margin(5)
        .x_label_area_size(25)
        .y_label_area_size(25)
        .build_cartesian_2d(0.0f32..1.0f32, 0.0f32..1.0f32)
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    chart
        .draw_series((0..NPLOTSAMPLES).flat_map(|x| {
            (0..NPLOTSAMPLES).map(move |y| {
                let pred = f64::from(pred[(x * NPLOTSAMPLES + y) as usize][0]);
                let c = COLORMAP.eval_continuous(pred);
                let px = x as f32 / NPLOTSAMPLES as f32;
                let py = y as f32 / NPLOTSAMPLES as f32;
                Rectangle::new(
                    [(px, py), (px + PW, py + PW)],
                    RGBColor(c.r, c.g, c.b).filled(),
                )
            })
        }))
        .unwrap();

    chart
        .draw_series(xs.iter().zip(ys.iter()).map(|([x, y], class)| {
            Circle::new(
                (*x, *y),
                3,
                if *class == 0f32 {
                    YELLOW.filled()
                } else {
                    BLUE.filled()
                },
            )
        }))
        .unwrap();

    root.present().unwrap();
}

fn main() {
    let (xs, ys) = generate_data(NSAMPLES);
    let dev = Cpu::default();

    let mut mlp = dev.build_module::<Mlp, f32>();

    let mut grads = mlp.alloc_grads();

    let mut opt = Adam::new(&mlp, AdamConfig::default());

    let x: Tensor<Rank2<NSAMPLES, 2>, f32, _> =
        dev.tensor(xs.iter().copied().flatten().collect::<Vec<f32>>());

    let y: Tensor<Rank2<NSAMPLES, 1>, f32, _> = dev.tensor(ys.clone());

    /*let (mlp, dev, grads, train_losses, test_losses) =
    train_on(&xs, &ys, &test_xs, &test_ys, NITERS);*/

    let meshgrid = get_meshgrid(&dev);

    /*graph_decision_map(&pred, &xs, &ys);

    graph_losses(&train_losses, "train_losses.png");
    graph_losses(&test_losses, "test_losses.png");*/

    let mut buf = BufferWrapper(vec![0u32; W * H]);

    let mut window = Window::new("MLNET - Demo", W, H, WindowOptions::default()).unwrap();
    //window.limit_update_rate(Some(std::time::Duration::from_micros(16600)));

    while window.is_open() && !window.is_key_down(Key::Escape) {
        mlp.zero_grads(&mut grads);

        let train_prediction = mlp.forward_mut(x.trace(grads));
        let train_loss = mse_loss(train_prediction, y.clone());

        grads = train_loss.backward();

        opt.update(&mut mlp, &grads)
            .expect("Oops, there were some unused params");

        let pred = mlp.forward(meshgrid.trace(grads.clone())).array();
        graph_decision_map(
            &pred,
            &xs,
            &ys,
            std::borrow::BorrowMut::borrow_mut(&mut buf),
        );

        window
            .update_with_buffer(std::borrow::Borrow::borrow(&buf), W, H)
            .unwrap();
    }
}
