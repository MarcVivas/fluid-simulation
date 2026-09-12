use clap::{Arg, Command, value_parser};
use gpu_fluid_simulation::app::App;
use gpu_fluid_simulation::world::particles::ParticleInitPreset;
use winit::event_loop::{ControlFlow, EventLoop};

fn main() -> anyhow::Result<()> {
    let args = Command::new(env!("CARGO_PKG_NAME"))
        .version(env!("CARGO_PKG_VERSION"))
        .about("GPU fluid simulation using Vulkan and Slang")
        .arg(
            Arg::new("num-particles")
                .long("num-particles")
                .value_name("COUNT")
                .help("Number of particles to simulate")
                .default_value("1000000")
                .value_parser(value_parser!(u32).range(1..)),
        )
        .arg(
            Arg::new("preset")
                .long("preset")
                .help("Initial particle arrangement")
                .default_value("colliding-blocks")
                .value_parser(["double-dam-break", "rotating-block", "colliding-blocks"]),
        )
        .get_matches();

    let num_particles = *args.get_one::<u32>("num-particles").unwrap() as usize;
    let preset = match args.get_one::<String>("preset").unwrap().as_str() {
        "double-dam-break" => ParticleInitPreset::DoubleDamBreak,
        "rotating-block" => ParticleInitPreset::RotatingBlock,
        "colliding-blocks" => ParticleInitPreset::CollidingBlocks,
        _ => unreachable!("preset values are checked by clap"),
    };

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new(num_particles, preset);
    event_loop.run_app(&mut app)?;
    Ok(())
}
