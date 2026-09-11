use glam::{Mat4, Quat, Vec2, Vec3, Vec4};
use winit::dpi::PhysicalSize;

/// Stores where you are looking at.
pub struct Camera {
    pub position: Vec3,
    pub target: Vec3,
    pub fov: f32,
    pub pitch: f32,
    pub yaw: f32,
    pub distance: f32,
    camera_uniform: CameraUniform,
    aspect_ratio: f32,
}

impl Camera {
    pub fn new(world_size: &Vec3, window_size: &PhysicalSize<u32>) -> Self {
        let target = Vec3::new(world_size.x / 2.0, world_size.y / 2.0, world_size.z / 2.0);

        // Isometric-ish setup
        let pitch = 60.0f32.to_radians();
        let yaw = 60.0f32.to_radians();

        // Distance calc
        let fov = 45.0f32.to_radians();
        let bounding_radius = (world_size.length() / 2.0) * 0.25;
        let distance = bounding_radius / (fov / 2.0).tan();

        // Initial position
        let rotation = Quat::from_euler(glam::EulerRot::YXZ, yaw, -pitch, 0.0);
        let offset = rotation * Vec3::new(0.0, 0.0, distance);
        let position = target + offset;

        let aspect_ratio = window_size.width as f32 / window_size.height as f32;
        let camera_uniform = CameraUniform::new();

        Self {
            position,
            target,
            fov,
            distance,
            pitch,
            yaw,
            aspect_ratio,
            camera_uniform,
        }
    }

    // Standard Matrix builders
    pub fn build_view_projection_matrix(&mut self, screen_size: &Vec2) -> (Mat4, Mat4) {
        self.aspect_ratio = screen_size.x / screen_size.y;
        let view = Mat4::look_at_rh(self.position, self.target, Vec3::Y);
        let near = 0.5;
        let far = self.distance * 10.0;
        let projection = Mat4::perspective_rh(self.fov, self.aspect_ratio, near, far);
        let correction = Mat4::from_cols(
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, -1.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.5, 0.0),
            Vec4::new(0.0, 0.0, 0.5, 1.0),
        );
        let final_proj = correction * projection;
        self.camera_uniform
            .update_view_projection(&view, &final_proj);
        (view, final_proj)
    }

    #[allow(unused)]
    pub fn screen_to_world(&self, screen_size: &Vec2, screen_pos: &Vec2) -> Option<Vec2> {
        let (view, proj) = self.get_matrices_for_calculation(screen_size);
        let view_proj_inv = (proj * view).inverse();
        let ndc_x = (screen_pos.x / screen_size.x) * 2.0 - 1.0;
        let ndc_y = (screen_pos.y / screen_size.y) * 2.0 - 1.0;
        let world_near = view_proj_inv * Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
        let world_far = view_proj_inv * Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
        let near_p = world_near.truncate() / world_near.w;
        let far_p = world_far.truncate() / world_far.w;
        let dir = (far_p - near_p).normalize();
        let denom = Vec3::Z.dot(dir); // Assuming Z-up plane for world interaction
        if denom.abs() < 1e-6 {
            return None;
        }
        let t = -(Vec3::Z.dot(near_p)) / denom;
        if t < 0.0 {
            return None;
        }
        let hit = near_p + dir * t;
        Some(Vec2::new(hit.x, hit.y))
    }
    fn get_matrices_for_calculation(&self, screen_size: &Vec2) -> (Mat4, Mat4) {
        let view = Mat4::look_at_rh(self.position, self.target, Vec3::Y);
        let aspect_ratio = screen_size.x / screen_size.y;
        let projection = Mat4::perspective_rh(self.fov, aspect_ratio, 0.5, self.distance * 10.0);
        let correction = Mat4::from_cols(
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, -1.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.5, 0.0),
            Vec4::new(0.0, 0.0, 0.5, 1.0),
        );
        (view, correction * projection)
    }
    pub fn get_uniform(&self) -> &CameraUniform {
        &self.camera_uniform
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CameraUniform {
    pub view: Mat4,
    pub projection: Mat4,
}
impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view: Mat4::IDENTITY,
            projection: Mat4::IDENTITY,
        }
    }
    pub fn update_view_projection(&mut self, view: &Mat4, projection: &Mat4) {
        self.view = view.transpose();
        self.projection = projection.transpose();
    }
}
